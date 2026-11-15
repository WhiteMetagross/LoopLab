"""Bounded BPI derived active work simulation, prediction, and tabular dispatch."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path
import platform
import random
import signal
import sys
import time
import xml.etree.ElementTree as ET

import joblib
import numpy as np
import scipy
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance
import sklearn
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from src.loop_feasibility import attributes, digest, local, timestamp, write_json


def work_segments(events, cfg):
    opened, result, counts = {}, [], Counter()
    invalid = False
    incomplete = False
    for event in sorted(events, key=lambda e: timestamp(e["time:timestamp"])):
        activity = event["concept:name"]
        incomplete |= activity == "A_Incomplete"
        if not activity.startswith("W_"):
            continue
        life = event.get("lifecycle:transition", "").lower()
        counts["life:"+life] += 1
        at = timestamp(event["time:timestamp"])
        if life in ("start", "resume"):
            if activity in opened:
                counts["ambiguous_duplicate_open"] += 1
                invalid = True
            opened[activity] = (at, life, incomplete, event.get("org:resource"))
        elif life in ("suspend", "complete", "ate_abort", "withdraw"):
            if activity not in opened:
                counts["close_without_active_start"] += 1
                continue
            start, start_life, marker, resource = opened.pop(activity)
            seconds = at-start
            if seconds > cfg["max_segment_minutes"] * 60:
                counts["segments_over_limit"] += 1
                invalid = True
            elif seconds >= cfg["minimum_segment_seconds"]:
                result.append({"activity": activity, "minutes": seconds/60, "new_visit": start_life == "start",
                               "incomplete_seen": marker, "start": start, "end": at, "resource": resource})
            else:
                counts["zero_or_tiny_segments"] += 1
    counts["unclosed_active_starts"] += len(opened)
    invalid |= bool(opened)
    result.sort(key=lambda s: s["start"])
    counts["matched_positive_segments"] = len(result)
    counts["cases_invalid"] = int(invalid)
    return result if not invalid else [], counts


def features(case):
    visits, consumed, resumed = Counter(), 0., 0
    result = []
    for i, segment in enumerate(case["segments"]):
        a = segment["activity"]
        visits[a] += int(segment["new_visit"])
        resumed += int(not segment["new_visit"])
        result.append({"activity": a, "position": i, "visit": visits[a],
                       "repeat_visit": int(visits[a] > 1), "resumption": int(not segment["new_visit"]),
                       "past_resume_count": resumed, "consumed_log_minutes": np.log1p(consumed),
                       "incomplete_seen": int(segment["incomplete_seen"]),
                       "last": case["segments"][i-1]["activity"] if i else "START",
                       **{"visits:"+key: value for key, value in visits.items()}})
        consumed += segment["minutes"]
    return result


def labels(case):
    durations = [s["minutes"] for s in case["segments"]]
    return np.cumsum(durations[::-1])[::-1]


def prepare(cfg):
    out = Path(cfg["out"])
    out.mkdir(parents=True, exist_ok=True)
    if (out / "data.joblib").exists():
        return joblib.load(out / "data.joblib")
    data, profile = {}, {}
    for name, spec in cfg["datasets"].items():
        split = json.loads(Path(spec["split"]).read_text())
        assignments = {case: part for part, key in (("train", "train_ids"), ("development", "validation_ids")) for case in split[key]}
        protected = set(split["test_ids"])
        assert not set(assignments) & protected
        portions, audit = {"train": [], "development": []}, Counter()
        with gzip.open(spec["file"], "rb") as stream:
            parser = ET.iterparse(stream, events=("start", "end"))
            _, root = next(parser)
            for kind, node in parser:
                if kind != "end" or local(node.tag) != "trace":
                    continue
                case_id = attributes(node).get("concept:name")
                if case_id in assignments:
                    events = [attributes(e) for e in node if local(e.tag) == "event"]
                    segments, counts = work_segments(events, cfg)
                    audit.update(counts)
                    part = assignments[case_id]
                    audit[part+":examined_cases"] += 1
                    if segments:
                        portions[part].append({"id": case_id, "segments": segments, "start": segments[0]["start"]})
                    else:
                        audit[part+":excluded_cases"] += 1
                node.clear()
                root.clear()
        for part, limit in (("train", cfg["max_train_cases"]), ("development", cfg["max_development_cases"])):
            cases = sorted(portions[part], key=lambda c: c["start"])
            if len(cases) > limit:
                selected = set(np.random.default_rng(cfg["seed"]).choice(len(cases), limit, replace=False))
                cases = [c for i, c in enumerate(cases) if i in selected]
            portions[part] = cases
        train = portions.pop("train")
        nfit = int(len(train)*cfg["ml_fit_share"])
        ncal = int(len(train)*cfg["ml_calibration_share"])
        portions.update(fit=train[:nfit], calibration=train[nfit:nfit+ncal], rl=train[nfit+ncal:])
        ids = {part: [c["id"] for c in cases] for part, cases in portions.items()}
        for key, values in ids.items():
            assert not set(values) & protected
            for other, compared in ids.items():
                if key != other:
                    assert not set(values) & set(compared)
        portions["ids"] = ids
        data[name] = portions
        profile[name] = {"audit": dict(audit), "case_counts": {p: len(v) for p, v in ids.items()},
                         "raw_sha256": digest(spec["file"]), "split_sha256": digest(spec["split"]),
                         "protected_overlap": 0, "final_scored": False}
        print(json.dumps({"dataset": name, **profile[name]}), flush=True)
    joblib.dump(data, out / "data.joblib", compress=3)
    write_json(out / "profile.json", profile)
    write_json(out / "config.json", cfg)
    return data


def lookup_key(row):
    return row["activity"], min(row["visit"], 3), row["resumption"]


def fit_predictors(parts, cfg):
    from catboost import CatBoostRegressor
    rows = [r for case in parts["fit"] for r in features(case)]
    y = np.concatenate([labels(case) for case in parts["fit"]])
    vector = DictVectorizer(sparse=False)
    X = vector.fit_transform(rows)
    scaler = StandardScaler().fit(X)
    cells, activity, service = defaultdict(list), defaultdict(list), defaultdict(list)
    for row, value in zip(rows, y):
        cells[lookup_key(row)].append(value)
        activity[row["activity"]].append(value)
    for case in parts["fit"]:
        for row, segment in zip(features(case), case["segments"]):
            service[(row["activity"], row["resumption"])].append(segment["minutes"])
    model = {"vector": vector, "scaler": scaler, "global": float(np.median(y)),
             "cells": {key: float(np.median(v)) for key, v in cells.items() if len(v) >= 20},
             "activity": {key: float(np.median(v)) for key, v in activity.items()},
             "service": {key: float(np.median(v)) for key, v in service.items()}}
    with threadpool_limits(limits=1):
        ridge = Ridge(alpha=10).fit(scaler.transform(X), np.log1p(y))
        cat = CatBoostRegressor(iterations=cfg["boost_iterations"], depth=4, learning_rate=.05, loss_function="RMSE",
                                random_seed=cfg["seed"], thread_count=1, verbose=False, allow_writing_files=False)
        cat.fit(X, np.log1p(y))
    model.update(Ridge=ridge, CatBoost=cat)
    scores = {}
    for part in ("calibration", "development"):
        rr = [r for case in parts[part] for r in features(case)]
        yy = np.concatenate([labels(case) for case in parts[part]])
        scores[part] = {}
        for family in ("global", "lookup", "Ridge", "CatBoost"):
            pred = predict(model, rr, family)
            scores[part][family] = {"MAE_minutes": float(mean_absolute_error(yy, pred)), "rows": len(yy)}
        # Family selection is fixed from calibration only, before development use.
        if part == "calibration":
            model["selected"] = min(scores[part], key=lambda f: scores[part][f]["MAE_minutes"])
    return model, scores


def predict(model, rows, family):
    if family == "global":
        return np.full(len(rows), model["global"])
    if family == "lookup":
        return np.array([model["cells"].get(lookup_key(r), model["activity"].get(r["activity"], model["global"])) for r in rows])
    X = model["vector"].transform(rows)
    if family == "Ridge":
        X = model["scaler"].transform(X)
    with threadpool_limits(limits=1):
        return np.maximum(.01, np.expm1(np.clip(model[family].predict(X), -10, 15)))


def enrich(cases, model, family):
    output = []
    for case in cases:
        rows = features(case)
        output.append({"id": case["id"], "duration": np.array([s["minutes"] for s in case["segments"]]),
                       "pred": predict(model, rows, family),
                       "service_est": np.array([model["service"].get((r["activity"], r["resumption"]), 1.) for r in rows]),
                       "repeat": np.array([r["repeat_visit"] for r in rows], dtype=int)})
    return output


class Queue:
    def __init__(self, cases, arrivals, scale):
        self.cases, self.arrivals, self.scale = cases, np.asarray(arrivals), scale
        self.position = np.zeros(len(cases), dtype=int)
        self.ready_since = self.arrivals.copy()
        self.finished = np.zeros(len(cases), dtype=bool)
        self.completed = np.zeros(len(cases))
        self.clock, self.area, self.steps = float(self.arrivals.min()), 0., 0

    def available(self):
        indices = np.flatnonzero(~self.finished & (self.arrivals <= self.clock))
        if not len(indices) and not self.finished.all():
            self.clock = float(self.arrivals[~self.finished].min())
            indices = np.flatnonzero(~self.finished & (self.arrivals <= self.clock))
        return indices

    def choices(self):
        ready = self.available()
        oldest = min(ready, key=lambda i: (self.ready_since[i], i))
        shortest = min(ready, key=lambda i: (self.cases[i]["service_est"][self.position[i]], self.ready_since[i], i))
        remaining = min(ready, key=lambda i: (self.cases[i]["pred"][self.position[i]], self.ready_since[i], i))
        repeated = min(ready, key=lambda i: (-self.cases[i]["repeat"][self.position[i]], self.ready_since[i], i))
        return [oldest, shortest, remaining, repeated]

    def state(self):
        ready = self.available()
        choices = self.choices()
        p = [self.cases[i]["pred"][self.position[i]] for i in ready]
        old = self.cases[choices[0]]["pred"][self.position[choices[0]]]
        repeat_share = np.mean([self.cases[i]["repeat"][self.position[i]] for i in ready])
        return (int(np.digitize(len(ready), [4, 9])), int(np.digitize(repeat_share, [.001, .5])),
                int(np.digitize(min(p)/max(old, .01), [.5, .8])), int(np.digitize(old/self.scale, [.5, 1.5])))

    def step(self, action):
        selected = self.choices()[action]
        duration = self.cases[selected]["duration"][self.position[selected]]
        stop = self.clock+duration
        arrived = ~self.finished & (self.arrivals <= self.clock)
        future = ~self.finished & (self.arrivals > self.clock) & (self.arrivals < stop)
        area = arrived.sum()*duration + np.sum(stop-self.arrivals[future])
        self.area += float(area)
        self.clock = float(stop)
        self.position[selected] += 1
        self.steps += 1
        if self.position[selected] == len(self.cases[selected]["duration"]):
            self.finished[selected] = True
            self.completed[selected] = stop
        else:
            self.ready_since[selected] = stop
        done = bool(self.finished.all())
        return -float(area)/(len(self.cases)*self.scale), done

    def metrics(self):
        assert self.finished.all()
        flow = self.completed-self.arrivals
        work = np.array([case["duration"].sum() for case in self.cases])
        assert np.isclose(flow.sum(), self.area)
        assert self.steps == sum(len(c["duration"]) for c in self.cases)
        return {"mean_flow_minutes": float(flow.mean()), "mean_wait_minutes": float((flow-work).mean()),
                "mean_work_minutes": float(work.mean()), "p90_flow_minutes": float(np.quantile(flow, .9)),
                "completed_cases": len(self.cases), "segments": self.steps,
                "makespan_minutes": float(self.completed.max()-self.arrivals.min())}


def workload(cases, scale, cfg, rng, load):
    chosen = rng.choice(len(cases), cfg["cases_per_episode"], replace=False)
    arrivals = np.cumsum(rng.exponential(scale/load, cfg["cases_per_episode"]))
    return [cases[i] for i in chosen], arrivals


def fit_q(cases, scale, cfg):
    rng = np.random.default_rng(cfg["seed"])
    q = defaultdict(lambda: np.zeros(4))
    history = []
    for episode in range(cfg["q_episodes"]):
        epsilon = cfg["epsilon_start"]+(cfg["epsilon_end"]-cfg["epsilon_start"])*episode/max(1, cfg["q_episodes"]-1)
        pool, arrivals = workload(cases, scale, cfg, rng, float(rng.choice(cfg["loads"])))
        env = Queue(pool, arrivals, scale)
        state = env.state()
        while True:
            action = int(rng.integers(4)) if rng.random() < epsilon else int(np.argmax(q[state]))
            reward, done = env.step(action)
            next_state = None if done else env.state()
            target = reward+(0 if done else cfg["gamma"]*np.max(q[next_state]))
            q[state][action] += cfg["alpha"]*(target-q[state][action])
            state = next_state
            if done:
                break
        if (episode+1) % 50 == 0:
            history.append({"episode": episode+1, "mean_flow_minutes": env.metrics()["mean_flow_minutes"]})
    return dict(q), history


def fidelity(fit, dev, seed):
    rng = np.random.default_rng(seed)
    bootstrap = [fit[i] for i in rng.integers(0, len(fit), len(dev))]
    def summaries(cases):
        totals, repeats, edges = [], [], Counter()
        for case in cases:
            seq = [s["activity"] for s in case["segments"]]
            visits = [s["activity"] for s in case["segments"] if s["new_visit"]]
            totals.append(sum(s["minutes"] for s in case["segments"]))
            repeats.append(len(visits)-len(set(visits)))
            edges.update(zip(["START"]+seq, seq+["END"]))
        return totals, repeats, edges
    a, ar, ae = summaries(bootstrap)
    b, br, be = summaries(dev)
    alphabet = sorted(set(ae) | set(be))
    return {"active_work_Wasserstein_minutes": float(wasserstein_distance(a, b)),
            "active_work_normalized_Wasserstein": float(wasserstein_distance(a, b)/np.mean(b)),
            "new_visit_repeat_Wasserstein": float(wasserstein_distance(ar, br)),
            "transition_JS_divergence_bits": float(jensenshannon([ae[k] for k in alphabet], [be[k] for k in alphabet], base=2)**2),
            "training_bootstrap_repeat_case_fraction": float(np.mean(np.array(ar) > 0)),
            "development_repeat_case_fraction": float(np.mean(np.array(br) > 0)),
            "training_active_work_quantiles": np.quantile(a, [.1, .5, .9]).tolist(),
            "development_active_work_quantiles": np.quantile(b, [.1, .5, .9]).tolist()}


def evaluate_policies(parts, model, q_tables, cfg, scale):
    variants = {family: enrich(parts["development"], model, family) for family in ("lookup", model["selected"])}
    rng = np.random.default_rng(cfg["seed"])
    reference = variants[model["selected"]]
    records, cohorts = [], []
    for load in cfg["loads"]:
        for batch in range(cfg["evaluation_batches_per_load"]):
            chosen = rng.choice(len(reference), cfg["cases_per_episode"], replace=False)
            arrivals = np.cumsum(rng.exponential(scale/load, cfg["cases_per_episode"]))
            cohorts.append({"load": load, "batch": batch, "case_indices": chosen.tolist(), "arrivals": arrivals.tolist()})
            policies = [("FIFO", 0, model["selected"]), ("SPT_estimated", 1, model["selected"]),
                        ("ML_remaining", 2, model["selected"]), ("repeat_first", 3, model["selected"]),
                        ("Q_lookup", None, "lookup"), ("Q_ML", None, model["selected"])]
            for policy, fixed, family in policies:
                env = Queue([variants[family][i] for i in chosen], arrivals, scale)
                table = q_tables[family]
                while True:
                    action = fixed if fixed is not None else int(np.argmax(table.get(env.state(), np.zeros(4))))
                    _, done = env.step(action)
                    if done:
                        break
                records.append({"load": load, "batch": batch, "policy": policy, **env.metrics()})
    return records, cohorts


def run(cfg):
    import catboost
    out = Path(cfg["out"])
    if (out / "results.json").exists():
        raise FileExistsError("Completed pilot exists.")
    started = time.perf_counter()
    data = prepare(cfg)
    report = {"seed": cfg["seed"], "final_scored": False, "source_sha256": digest(__file__),
              "versions": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
                           "sklearn": sklearn.__version__, "catboost": catboost.__version__},
              "runtime": {"prefix": sys.prefix, "platform": platform.platform(), "device": "CPU"}, "datasets": {}}
    for name, parts in data.items():
        at = time.perf_counter()
        model, scores = fit_predictors(parts, cfg)
        scale = float(np.mean([labels(c)[0] for c in parts["rl"]]))
        q_tables, histories = {}, {}
        for family in dict.fromkeys(("lookup", model["selected"])):
            q_tables[family], histories[family] = fit_q(enrich(parts["rl"], model, family), scale, cfg)
        records, cohorts = evaluate_policies(parts, model, q_tables, cfg, scale)
        result = {"prediction": scores, "selected_predictor": model["selected"], "training_mean_work_minutes": scale,
                  "fidelity": fidelity(parts["fit"], parts["development"], cfg["seed"]), "policy_records": records,
                  "Q_training_history": histories, "Q_states": {k: len(v) for k, v in q_tables.items()},
                  "seconds": time.perf_counter()-at}
        report["datasets"][name] = result
        joblib.dump({"model": model, "q": q_tables, "scale": scale, "cohorts": cohorts}, out / (name+"_models.joblib"))
        write_json(out / "results.json", report)
        print(json.dumps({"dataset": name, "selected": model["selected"], "prediction": scores,
                          "fidelity": result["fidelity"], "seconds": result["seconds"]}), flush=True)
    report["wall_seconds"] = time.perf_counter()-started
    write_json(out / "results.json", report)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/bpi_queue_pilot.json")
    p.add_argument("--prepare-only", action="store_true")
    args = p.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    def timeout(signum, frame):
        raise TimeoutError("Bounded study wall cap reached.")
    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(cfg["stage_wall_seconds"])
    if args.prepare_only:
        prepare(cfg)
    else:
        run(cfg)


if __name__ == "__main__":
    from src.bpi_queue import main as canonical_main
    canonical_main()
