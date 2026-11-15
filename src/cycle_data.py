"""Observed incomplete handling episodes and native repetitive review strands."""
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path

import joblib
import numpy as np

from src.bpi_queue import features as base_features
from src.simbank_support import HQEnvironment, bank_action, run_policy

CLOSURES = {"A_Incomplete", "A_Pending", "A_Cancelled", "A_Denied"}


def bpi_case(case, events):
    episodes, opened, seen, audit = [], None, [], Counter()
    for event in events:
        name, when = event["activity"], event["time"]
        if name == "A_Validating" and opened is not None:
            opened["revalidated"] = True
        if name in CLOSURES and opened is not None:
            episodes.append({**opened, "end": when, "outcome": name})
            opened = None
        if name == "A_Incomplete":
            opened = {"begin": when, "revalidated": False}
            seen.append(when)
    if opened is not None:
        audit["excluded_unclosed_marker_case"] += 1
        return None, audit
    starts = np.array([s["start"] for s in case["segments"]])
    valid, cycle = [], np.full(len(starts), -1, dtype=int)
    for episode in episodes:
        first = int(np.searchsorted(starts, episode["begin"], side="left"))
        last = int(np.searchsorted(starts, episode["end"], side="left")) - 1
        audit["closed_marker_intervals"] += 1
        audit["closed_intervals_revalidated"] += episode["revalidated"]
        audit["outcome:" + episode["outcome"]] += 1
        if first > last:
            audit["zero_schedulable_work_intervals"] += 1
            continue
        assert np.all(cycle[first:last + 1] == -1)
        cycle[first:last + 1] = len(valid)
        valid.append({**episode, "first": first, "last": last})
    result = {**deepcopy(case), "cycle": cycle, "episodes": valid,
              "markers_seen": [int(np.searchsorted(seen, start, side="right")) for start in starts],
              "context": {}}
    audit["retained_cases"] += 1
    audit["schedulable_episodes"] += len(valid)
    return result, audit


def episode_features(case):
    basic = base_features(case)
    result = []
    for i, row in enumerate(basic):
        current = int(case["cycle"][i])
        completed = [episode for episode in case["episodes"] if episode["last"] < i]
        consumed = sum(case["segments"][j]["minutes"] for j in range(i) if case["cycle"][j] == current) if current >= 0 else 0.
        previous = sum(s["minutes"] for s in case["segments"][completed[-1]["first"]:completed[-1]["last"] + 1]) if completed else 0.
        result.append({**row, **case["context"], "episode:active": int(current >= 0),
                       "episode:markers_seen": case["markers_seen"][i], "episode:completed": len(completed),
                       "episode:consumed_log": float(np.log1p(consumed)),
                       "episode:previous_effort_log": float(np.log1p(previous))})
    return result


def targets(case):
    durations = np.array([s["minutes"] for s in case["segments"]])
    result = {name: [] for name in ["another_return", "remaining_cycles", "future_episode_work", "current_episode_work", "case_work"]}
    for i in range(len(durations)):
        active = int(case["cycle"][i])
        result["another_return"].append(int(any(e["first"] > i for e in case["episodes"])))
        result["remaining_cycles"].append(sum(e["last"] >= i for e in case["episodes"]))
        result["future_episode_work"].append(float(durations[i:][case["cycle"][i:] >= 0].sum()))
        result["current_episode_work"].append(float(sum(durations[j] for j in range(i, len(durations)) if case["cycle"][j] == active)) if active >= 0 else 0.)
        result["case_work"].append(float(durations[i:].sum()))
    return {key: np.array(value) for key, value in result.items()}


def analytic_simbank(case):
    contact = 864. if case["context"]["is_call"] else 1152.
    validation = 1008.
    result = {name: [] for name in targets(case)}
    for i, segment in enumerate(case["segments"]):
        previous_contacts = sum(s["activity"] in ("email_customer", "call_customer") for s in case["segments"][:i])
        is_contact = segment["activity"] in ("email_customer", "call_customer")
        initial = i == 0
        contacts_left = 3 - previous_contacts
        validations_left = contacts_left if is_contact or initial else contacts_left + 1
        episode_work = contacts_left * contact + validations_left * validation
        result["another_return"].append(int(contacts_left - int(is_contact) > 0))
        result["remaining_cycles"].append(validations_left)
        result["future_episode_work"].append(episode_work)
        result["current_episode_work"].append(contact + validation if is_contact else (0. if initial else validation))
        result["case_work"].append(episode_work + (validation if initial else 0.))
    return {key: np.array(value) for key, value in result.items()}


def simbank_cases(config):
    output, profile = {}, {}
    env = HQEnvironment()
    for part, requested in config["simbank_cases"].items():
        cases, attempted = [], 0
        while len(cases) < requested:
            scenario = config["seed"] + config["simbank_seed_offsets"][part] + attempted
            result, _ = run_policy(env, scenario, lambda state: bank_action(state, env.params))
            trace = env.generator.end_simulation_inference()
            attempted += 1
            if any(event["activity"] == "start_priority" for event in trace):
                continue
            chosen = [event for event in trace if event["activity"] in ("validate_application", "email_customer", "call_customer")]
            segments, episodes, current = [], [], -1
            for i, event in enumerate(chosen):
                activity = event["activity"]
                if activity in ("email_customer", "call_customer"):
                    episodes.append({"first": i, "last": i + 1, "revalidated": True, "outcome": "information_gathering"})
                    current = len(episodes) - 1
                minutes = env.generator.activity_executioner.times_dic[activity] / 60
                segments.append({"activity": activity, "minutes": minutes, "new_visit": True,
                                 "incomplete_seen": False, "start": i, "end": i + 1, "resource": "reviewer"})
            assert len(episodes) == 3 and len(segments) == 7
            ids = np.array([-1, 0, 0, 1, 1, 2, 2])
            case = {"id": str(scenario), "segments": segments, "cycle": ids, "episodes": episodes,
                    "markers_seen": [max(0, int(value) + 1) for value in ids],
                    "context": {"is_call": int(trace[0]["amount"] > 50000)}}
            for key, value in analytic_simbank(case).items():
                assert np.allclose(value, targets(case)[key])
            cases.append(case)
        output[part] = cases
        profile[part] = {"cases": len(cases), "generated_native_cases": attempted,
                         "priority_cases_outside_strand_population": attempted - len(cases),
                         "cycles": 3 * len(cases), "cycle_type": "repetitive information gathering, not corrective"}
    return output, profile


def prepare(config):
    out = Path(config["out"])
    out.mkdir(parents=True, exist_ok=True)
    if (out / "data.joblib").exists():
        raise FileExistsError("Prepared cycle data already exist.")
    original = joblib.load(Path(config["bpi_parent"]) / "data.joblib")
    markers = joblib.load(config["bpi_marker_cache"])
    bpi, bpi_profile = {}, {}
    for part in ["fit", "calibration", "rl", "development"]:
        cases, audit = [], Counter()
        for case in original["bpi2017"][part]:
            annotated, counts = bpi_case(case, markers[case["id"]])
            audit.update(counts)
            if annotated is not None:
                cases.append(annotated)
        bpi[part], bpi_profile[part] = cases, dict(audit)
    simbank, simbank_profile = simbank_cases(config)
    output = {"bpi2017": bpi, "simbank_repeat": simbank}
    profile = {"bpi2017": bpi_profile, "simbank_repeat": simbank_profile,
               "bpi2012_explicit_incomplete_seen_segments": sum(s["incomplete_seen"] for part in ["fit", "calibration", "rl", "development"]
                                                               for case in original["bpi2012"][part] for s in case["segments"]),
               "final_scored": False}
    joblib.dump(output, out / "data.joblib", compress=3)
    (out / "preparation.json").write_text(json.dumps(profile, indent=2) + "\n")
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(json.dumps(profile, indent=2), flush=True)
