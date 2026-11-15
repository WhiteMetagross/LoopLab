"""Bounded development checks. Repetition proxies are not causal rework labels."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import signal
import subprocess
import time
import xml.etree.ElementTree as ET

import joblib
import numpy as np
import scipy
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, precision_score, recall_score, roc_auc_score, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits


def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, default=str) + "\n")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def local(tag):
    return tag.rsplit("}", 1)[-1]


def attributes(element):
    return {x.get("key"): x.get("value") for x in element if x.get("key")}


def timestamp(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()


def load_log(path, include_case_attributes=False):
    traces, schema, lifecycles = [], Counter(), Counter()
    raw_events = missing = tied = unsorted = 0
    with gzip.open(path, "rb") as stream:
        parser = ET.iterparse(stream, events=("start", "end"))
        _, root = next(parser)
        for action, element in parser:
            if action != "end" or local(element.tag) != "trace":
                continue
            case = attributes(element)
            events = []
            for e in element:
                if local(e.tag) != "event":
                    continue
                a = attributes(e)
                schema.update(a.keys())
                raw_events += 1
                life = a.get("lifecycle:transition", "MISSING").lower()
                lifecycles[life] += 1
                if not a.get("concept:name") or not a.get("time:timestamp"):
                    missing += 1
                    continue
                events.append((a["concept:name"], timestamp(a["time:timestamp"]), life))
            if events:
                unsorted += sum(events[i][1] < events[i-1][1] for i in range(1, len(events)))
                events.sort(key=lambda x: x[1])  # Stable XML order breaks timestamp ties.
                tied += sum(events[i][1] == events[i-1][1] for i in range(1, len(events)))
                complete = [(a, t) for a, t, life in events if life in ("complete", "missing")]
                traces.append({"id": case.get("concept:name", str(len(traces))),
                               "start": events[0][1], "end": events[-1][1],
                               "raw": [(a, t) for a, t, _ in events], "events": complete})
                if include_case_attributes:
                    traces[-1]["case_attrs"] = case
            element.clear()
            root.clear()
    return traces, {"raw_events": raw_events, "event_attribute_counts": schema,
                    "lifecycle_counts": lifecycles, "missing_name_or_timestamp": missing,
                    "timestamp_ties_raw": tied, "negative_timestamp_steps_raw": unsorted}


def view_stats(traces, key):
    sequences = [[a for a, _ in x[key]] for x in traces]
    counts = Counter(a for seq in sequences for a in seq)
    repeated = [len(s) - len(set(s)) for s in sequences]
    self_loops = [sum(a == b for a, b in zip(s, s[1:])) for s in sequences]
    lengths = np.array([len(s) for s in sequences])
    variants = Counter(tuple(s) for s in sequences)
    nonempty = int(sum(lengths > 0))
    return {"cases": len(traces), "nonempty_cases": nonempty, "events": int(lengths.sum()),
            "activities": len(counts), "length_quantiles_0_50_90_99_100": np.quantile(lengths, [0,.5,.9,.99,1]).tolist(),
            "cases_with_repetition": sum(x > 0 for x in repeated),
            "cases_with_self_loop": sum(x > 0 for x in self_loops),
            "repeated_events": int(sum(repeated)), "self_loop_events": int(sum(self_loops)),
            "unique_activity_variants": len(variants), "top_activities": counts.most_common(30),
            "top_variants": [{"activities": list(s), "count": n} for s,n in variants.most_common(8)]}


def semantic_stats(traces):
    rejected_cases = resubmitted = reject_events = 0
    repeated_by_activity, transitions = Counter(), Counter()
    for tr in traces:
        seq = [a for a,_ in tr["events"]]
        c = Counter(seq)
        repeated_by_activity.update({a: n-1 for a,n in c.items() if n > 1})
        transitions.update(zip(seq, seq[1:]))
        rejected = [i for i,a in enumerate(seq) if "reject" in a.lower()]
        rejected_cases += bool(rejected)
        reject_events += len(rejected)
        resubmitted += any(any("submitted" in b.lower() for b in seq[i+1:]) for i in rejected)
    return {"cases_with_rejection_name": rejected_cases, "rejection_named_events": reject_events,
            "cases_with_rejection_then_later_submission_name": resubmitted,
            "repeated_activity_excess": repeated_by_activity.most_common(30),
            "top_direct_transitions": [[a,b,n] for (a,b),n in transitions.most_common(25)]}


def prefix_features(events, k, loop=False):
    seq = [a for a,_ in events[:k]]
    times = [t for _,t in events[:k]]
    counts = Counter(seq)
    features = {"length": k, "elapsed_log_hours": float(np.log1p((times[-1]-times[0])/3600)),
                "last_gap_log_hours": float(np.log1p((times[-1]-times[-2])/3600)) if k > 1 else 0.0}
    features.update({"count:"+a: float(n) for a,n in counts.items()})
    features.update({"last_"+str(j): a for j,a in enumerate(reversed(seq[-3:]))})
    if loop:
        seen, spans = {}, []
        for i,a in enumerate(seq):
            if a in seen:
                spans.append(i-seen[a])
            seen[a] = i
        prior = [i for i,a in enumerate(seq[:-1]) if a == seq[-1]]
        run = 1
        for a in reversed(seq[:-1]):
            if a != seq[-1]:
                break
            run += 1
        features.update({"loop:total_repeats": k-len(counts), "loop:last_visits": counts[seq[-1]],
                         "loop:self_run": run, "loop:last_return_gap": k-1-prior[-1] if prior else 0,
                         "loop:max_span": max(spans, default=0), "loop:mean_span": float(np.mean(spans)) if spans else 0,
                         "loop:repeat_last3": sum(seq[i] in seq[:i] for i in range(max(0,k-3), k))})
    return features


def make_rows(traces, prefixes, horizon):
    rows = []
    for tr in traces:
        events = tr["events"]
        for k in prefixes:
            if len(events) <= k:  # Conditional on case surviving this observation point.
                continue
            seq = [a for a,_ in events]
            future = seq[k:k+horizon]
            seen = set(seq[:k])
            rows.append({"case": tr["id"], "k": k, "activity": seq[k-1],
                         "visits": min(3, seq[:k].count(seq[k-1])),
                         "y": int(any(a in seen for a in future)),
                         "return_y": int(seq[k-1] in future),
                         "base": prefix_features(events,k), "loop": prefix_features(events,k,True)})
    return rows


def metrics(y,p):
    p = np.clip(p, 1e-6, 1-1e-6)
    binary = p >= .5
    return {"n": len(y), "positives": int(np.sum(y)), "prevalence": float(np.mean(y)),
            "average_precision": float(average_precision_score(y,p)),
            "roc_auc": float(roc_auc_score(y,p)) if len(set(y)) == 2 else None,
            "brier": float(brier_score_loss(y,p)), "log_loss": float(log_loss(y,p,labels=[0,1])),
            "f1_at_0.5": float(f1_score(y,binary,zero_division=0)),
            "precision_at_0.5": float(precision_score(y,binary,zero_division=0)),
            "recall_at_0.5": float(recall_score(y,binary,zero_division=0))}


def table_fit(rows, stateful):
    """Beta shrinkage to activity mean. Analytical fit; no optimizer."""
    totals, activity = defaultdict(lambda:[0,0]), defaultdict(lambda:[0,0])
    for r in rows:
        key = (r["activity"],r["visits"]) if stateful else (r["activity"],)
        totals[key][0] += r["return_y"]
        totals[key][1] += 1
        activity[r["activity"]][0] += r["return_y"]
        activity[r["activity"]][1] += 1
    prior = (sum(r["return_y"] for r in rows)+1)/(len(rows)+2)
    result = {"stateful":stateful,"prior":prior,"activity":{},"states":{}}
    for a,(s,n) in activity.items():
        result["activity"][a] = (s+2*prior)/(n+2)
    for key,(s,n) in totals.items():
        result["states"][json.dumps(key)] = (s+10*result["activity"][key[0]])/(n+10)
    return result


def table_predict(model, rows):
    return np.array([model["states"].get(json.dumps((r["activity"],r["visits"]) if model["stateful"] else (r["activity"],)),
                      model["activity"].get(r["activity"],model["prior"])) for r in rows])


def pilot(traces, cfg, out):
    ordered = sorted(traces, key=lambda t:(t["start"],t["id"]))
    n = len(ordered)
    cut1, cut2 = int(n*cfg["train_fraction"]), int(n*(cfg["train_fraction"]+cfg["validation_fraction"]))
    val_start, test_start = ordered[cut1]["start"], ordered[cut2]["start"]
    train = [x for x in ordered[:cut1] if x["end"] < val_start]
    val = [x for x in ordered[cut1:cut2] if x["end"] < test_start]
    # Last 20 percent never encoded, fitted, or scored. Metadata is descriptive only.
    sample_cap = cfg["max_cases_per_dataset"]
    def thin(items, limit):
        return [items[i] for i in np.linspace(0,len(items)-1,min(len(items),limit),dtype=int)] if items else []
    train = thin(train,int(sample_cap*.75))
    val = thin(val,int(sample_cap*.25))
    write_json(out/"split.json", {"train_ids":[x["id"] for x in train],"validation_ids":[x["id"] for x in val],
               "test_ids":[x["id"] for x in ordered[cut2:]],"test_scored":False,
               "validation_start":val_start,"test_start":test_start,
               "train_purged_overlap":cut1-sum(x["end"] < val_start for x in ordered[:cut1]),
               "validation_purged_overlap":cut2-cut1-sum(x["end"] < test_start for x in ordered[cut1:cut2])})
    assert not set(t["id"] for t in train) & set(t["id"] for t in val)
    assert max(x["end"] for x in train) < min(x["start"] for x in val)
    train_rows = make_rows(train,cfg["prefix_lengths"],cfg["future_event_horizon"])
    val_rows = make_rows(val,cfg["prefix_lengths"],cfg["future_event_horizon"])
    train_variants = {tuple(a for a,_ in t["events"]) for t in train}
    audit = {"train_cases":len(train),"validation_cases":len(val),"train_rows":len(train_rows),"validation_rows":len(val_rows),
             "validation_cases_with_activity_variant_seen_in_train":sum(tuple(a for a,_ in t["events"]) in train_variants for t in val),
             "same_case_overlap":0,"chronological_overlap_after_purge":False,
             "features": "prefix only; IDs and case payload excluded; vectorizer and scaler train only",
             "censoring": "Trace end treated as observed end, administrative censoring remains unverified.",
             "survival_condition": "Only cases with an event after prefix k are scored; not deployment eligibility evidence."}
    write_json(out/"audit.json",audit)
    results=[]
    y = np.array([r["y"] for r in val_rows])
    p0 = np.mean([r["y"] for r in train_rows])
    results.append({"task":"future_repetition_5_events","model":"constant_prevalence","seed":None,"fit_seconds":0,
                    "metrics":metrics(y,np.full(len(y),p0))})
    def timeout_handler(signum, frame):
        raise TimeoutError("Per model CPU smoke budget exceeded")
    signal.signal(signal.SIGALRM,timeout_handler)
    for seed in cfg["seed_list"]:
        random.seed(seed)
        np.random.seed(seed)
        for family in ("LR","RF"):
            for view in ("base","loop"):
                estimator = LogisticRegression(C=1.0,max_iter=300,random_state=seed) if family == "LR" else RandomForestClassifier(
                    n_estimators=cfg["forest_trees"],max_depth=cfg["forest_max_depth"],min_samples_leaf=10,n_jobs=1,random_state=seed)
                model = make_pipeline(DictVectorizer(sparse=False),StandardScaler(),estimator)
                started=time.perf_counter()
                signal.alarm(cfg["max_fit_seconds"])
                try:
                    with threadpool_limits(limits=1):
                        model.fit([r[view] for r in train_rows],[r["y"] for r in train_rows])
                        elapsed=time.perf_counter()-started
                        p=model.predict_proba([r[view] for r in val_rows])[:,1]
                    joblib.dump(model,out/f"{family}_{view}_{seed}.joblib")
                    np.savez_compressed(out/f"{family}_{view}_{seed}_validation.npz",y=y,p=p)
                    results.append({"task":"future_repetition_5_events","model":f"{family}_{view}","seed":seed,"fit_seconds":elapsed,"metrics":metrics(y,p)})
                except TimeoutError as error:
                    results.append({"task":"future_repetition_5_events","model":f"{family}_{view}","seed":seed,"error":str(error)})
                finally:
                    signal.alarm(0)
    for stateful in (False,True):
        started=time.perf_counter()
        table=table_fit(train_rows,stateful)
        elapsed=time.perf_counter()-started
        p=table_predict(table,val_rows)
        label="activity_visit_table" if stateful else "activity_only_table"
        write_json(out/f"{label}.json",table)
        yy=np.array([r["return_y"] for r in val_rows])
        np.savez_compressed(out/f"{label}_validation.npz",y=yy,p=p)
        results.append({"task":"current_activity_returns_5_events","model":label,"seed":None,"fit_seconds":elapsed,"metrics":metrics(yy,p),
                        "state_count":len(table["states"])})
    return results


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--config",default="configs/feasibility.json")
    parser.add_argument("--out",default="runs/feasibility_20261029")
    parser.add_argument("--profile-only",action="store_true")
    args=parser.parse_args()
    cfg=json.loads(Path(args.config).read_text())
    out=Path(args.out)
    out.mkdir(parents=True,exist_ok=True)
    write_json(out/"config.json",cfg)
    write_json(out/"environment.json",{"python":platform.python_version(),"platform":platform.platform(),"machine":platform.machine(),
        "numpy":np.__version__,"scipy":scipy.__version__,"sklearn":sklearn.__version__,"joblib":joblib.__version__,
        "conda_env":os.environ.get("CONDA_DEFAULT_ENV"),"cuda_used":False,"framework":"scikit-learn CPU",
        "git_commit":None,"git_note":"Workspace has no Git repository; source SHA256 recorded.",
        "source_sha256":digest(__file__),"config_sha256":digest(args.config),"utc":datetime.now(timezone.utc).isoformat()})
    for name,filename in cfg["datasets"].items():
        path=Path(filename)
        if not path.exists():
            print(json.dumps({"dataset":name,"status":"missing"}),flush=True)
            continue
        started=time.perf_counter()
        traces,metadata=load_log(path)
        stats={"sha256":digest(path),"downloaded_bytes":path.stat().st_size,**metadata,
               "raw":view_stats(traces,"raw"),"complete":view_stats(traces,"events"),"semantics":semantic_stats(traces),
               "first_event_utc":datetime.fromtimestamp(min(t["start"] for t in traces),timezone.utc).isoformat(),
               "last_event_utc":datetime.fromtimestamp(max(t["end"] for t in traces),timezone.utc).isoformat()}
        write_json(out/name/"profile.json",stats)
        print(json.dumps({"dataset":name,"parse_seconds":time.perf_counter()-started,"raw_events":metadata["raw_events"],
                         "complete_events":stats["complete"]["events"],"cases":len(traces),"semantics":stats["semantics"]}),flush=True)
        if not args.profile_only:
            results=pilot(traces,cfg,out/name)
            write_json(out/name/"metrics.json",results)
            print(json.dumps({"dataset":name,"results":results}),flush=True)


if __name__ == "__main__":
    main()
