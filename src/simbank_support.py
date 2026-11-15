"""Audited release loader and thin adapter around unchanged SimBank core."""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import pickle
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SAFE_GLOBALS = {
    ("builtins", "slice"), ("datetime", "datetime"),
    ("numpy", "dtype"), ("numpy", "ndarray"),
    ("numpy.core.multiarray", "_reconstruct"),
    ("pandas._libs.internals", "_unpickle_block"),
    ("pandas.core.frame", "DataFrame"),
    ("pandas.core.indexes.base", "Index"),
    ("pandas.core.indexes.base", "_new_Index"),
    ("pandas.core.indexes.range", "RangeIndex"),
    ("pandas.core.internals.managers", "BlockManager"),
}


class ReleasedUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) not in SAFE_GLOBALS:
            raise pickle.UnpicklingError(f"Unapproved global: {module}.{name}")
        return getattr(importlib.import_module(module), name)

    def persistent_load(self, pid):
        raise pickle.UnpicklingError("External pickle reference denied.")


def file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_released(path: str | Path):
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    manifest = json.loads((ROOT / "configs/simbank_release.json").read_text(encoding="utf-8"))
    record = manifest["parameter_file"]
    if path.name != record["name"]:
        raise ValueError(f"Unpinned SimBank object: {path.name}")
    if file_hash(path) != record["sha256"]:
        raise ValueError("Released object differs from acquisition checksum.")
    inspected = record["pickle_scan"]
    if inspected["sha256"] != record["sha256"]:
        raise ValueError("Object requires an opcode inspection first.")
    if not {tuple(item) for item in inspected["globals"]} <= SAFE_GLOBALS:
        raise ValueError("Inspected object contains unapproved globals.")
    with path.open("rb") as stream:
        return ReleasedUnpickler(stream).load()


def release_parameters():
    params = load_released("data/simbank/released/loan_log_['time_contact_HQ']_100000_dataset_params")
    params = deepcopy(params)
    params["intervention_info"]["RCT"] = False
    return params


def generator_class():
    folder = ROOT / "external/SimBank/SimBank"
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))
    return importlib.import_module("simulation").PresProcessGenerator


FEATURES = ["amount", "est_quality", "unc_quality", "elapsed_time", "cum_cost",
            "contact_count", "validation_count", "refusal_count", "prefix_length",
            "decision_index", "last_activity", "max_activity_count", "repeat_fraction"]


def prefix_features(prefix: list[dict], decision_index: int) -> dict:
    if not prefix:
        raise ValueError("Empty HQ decision prefix.")
    last = prefix[-1]
    counts = Counter(event["activity"] for event in prefix)
    result = {key: float(last[key]) for key in
              ["amount", "est_quality", "unc_quality", "elapsed_time", "cum_cost"]}
    result.update(contact_count=counts["email_customer"] + counts["call_customer"],
                  validation_count=counts["validate_application"], refusal_count=counts["receive_refusal"],
                  prefix_length=len(prefix), decision_index=decision_index,
                  last_activity=last["activity"], max_activity_count=max(counts.values()),
                  repeat_fraction=(len(prefix) - len(counts)) / len(prefix))
    assert set(result) == set(FEATURES)
    return result


class HQEnvironment:
    """Each reset uses a fresh explicit scenario seed and the release start date."""
    def __init__(self, params=None):
        self.params = deepcopy(params if params is not None else release_parameters())
        self.generator = generator_class()(deepcopy(self.params), seed=11)
        self.decision_index = 0
        self.branches = []

    def reset(self, scenario_seed: int):
        self.generator.random_seed = int(scenario_seed)
        self.generator.simulation_start = self.params["simulation_start"]
        self.decision_index = 0
        self.branches = self.generator.start_simulation_inference(seed_to_add=0)
        return self.observe()

    def observe(self):
        if not self.generator.int_points_available:
            return None
        if len(self.branches) != 2:
            raise ValueError("Expected wait and contact HQ branches.")
        # API exposes one post action event per candidate. Never feed it to ML.
        first = prefix_features(self.branches[0][:-1], self.decision_index)
        second = prefix_features(self.branches[1][:-1], self.decision_index)
        if first != second:
            raise ValueError("Candidate branches disagree on the observed prefix.")
        return first

    def step(self, action: int):
        if action not in (0, 1):
            raise ValueError("HQ action must be wait (0) or contact (1).")
        if not self.generator.int_points_available:
            raise ValueError("Cannot act after termination.")
        self.branches = self.generator.continue_simulation_inference(action)
        self.decision_index += 1
        if self.decision_index > 20:
            raise ValueError("HQ decision safety bound exceeded.")
        # Paper's four informative timings are raw indices 0, 2, 4, 6.
        # Validation alone changes no customer information. Force wait there.
        while self.generator.int_points_available and self.decision_index % 2:
            self.branches = self.generator.continue_simulation_inference(0)
            self.decision_index += 1
        return self.observe()

    def result(self) -> dict:
        if self.generator.int_points_available:
            raise ValueError("Case unfinished.")
        trace = self.generator.end_simulation_inference()
        last = trace[-1]
        counts = Counter(event["activity"] for event in trace)
        if not np.isfinite(last["outcome"]):
            raise ValueError("Nonfinite terminal profit.")
        if counts["contact_headquarters"] > 1:
            raise ValueError("Repeated HQ contact unsupported.")
        return {"profit": float(last["outcome"]), "elapsed_days": float(last["elapsed_time"]),
                "cost": float(last["cum_cost"]), "accepted": int(last["activity"] == "receive_acceptance"),
                "cancelled": int(last["activity"] == "cancel_application"),
                "contacts": counts["email_customer"] + counts["call_customer"],
                "validations": counts["validate_application"], "events": len(trace),
                "HQ_contacts": counts["contact_headquarters"], "decisions": self.decision_index,
                "last_activity": last["activity"],
                "activity_sequence": [event["activity"] for event in trace]}


def bank_action(state: dict, params: dict) -> int:
    policy = params["policies_info"]
    return int(state["contact_count"] < 2 and state["unc_quality"] == 0 and
               state["amount"] > 10000 and state["est_quality"] >= policy["min_quality"])


def run_policy(env: HQEnvironment, scenario_seed: int, policy) -> tuple[dict, list]:
    state = env.reset(scenario_seed)
    trajectory = []
    while state is not None:
        action = int(policy(state))
        trajectory.append((state, action))
        state = env.step(action)
    return env.result(), trajectory
