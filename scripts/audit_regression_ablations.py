"""Audit a completed Regression focused ablation run."""

import argparse
import json
from pathlib import Path

import numpy as np

from src.benchmark_models import seeds
from src.domain_loop_data import sha256
from src.domain_loop_simulations import save


EXPECTED = {
    "insurance": {"full", "equal_static", "no_congestion", "no_forecast", "no_loop_state", "first_only"},
    "bpisim": {"marker_ML", "marker_selected", "marker_forced", "marker_equal_static",
                "ready_ML", "ready_selected", "ready_forced", "ready_equal_static"},
    "manufacturing": {"ML", "raw", "mean_blend", "safe_050", "safe_075", "safe_100",
                      "no_accrued_wait", "mean_queue_objective", "burst_ML", "burst_raw", "burst_safe_075"},
    "simbank": {"Bank", "Regression_DP_500", "CQL_profit", "CQL_cancel", "CQL_refuse", "CQL_both",
                "Profit_FittedQ", "Profit_FittedQ_ensemble", "CQL_history", "Unconstrained_selector",
                "Constrained_selector", "logging_CQL_random", "logging_CQL_biased"},
}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def audit(root):
    manifest = read(root / "manifest.json")
    config, mode = manifest["config"], manifest["mode"]
    profile = config[mode]
    expected_seeds = seeds(config["master_seed"], config["replicates"])
    if mode == "smoke":
        expected_seeds = expected_seeds[:1]
    if manifest["seeds"] != expected_seeds:
        raise ValueError("Manifest seed list was not generated from master seed 17")
    status = read(root / "status.json")
    if status["stage"] != "complete" or status.get("failures"):
        raise ValueError("Run status is not cleanly complete")
    checks = {}
    for domain, required in EXPECTED.items():
        checks[domain] = []
        for index, seed in enumerate(expected_seeds):
            folder = root / domain / f"seed_{index}"
            complete = read(folder / "complete.json")
            design = read(folder / "design.json")
            if complete["seed"] != seed or design["seed"] != seed:
                raise ValueError(f"{domain} seed metadata mismatch")
            if design["forecast_family"] != "Regression":
                raise ValueError(f"{domain} did not hold Regression fixed")
            for path, digest in design["source_sha256"].items():
                if sha256(Path(path).read_bytes()) != digest:
                    raise ValueError(f"{domain} source hash changed: {path}")
            with np.load(folder / "paired.npz") as arrays:
                names = set(arrays.files)
                missing = required - names
                if missing:
                    raise ValueError(f"{domain} missing arms: {sorted(missing)}")
                for name in arrays.files:
                    values = arrays[name]
                    if values.ndim != 2 or values.shape[1] != len(design["fields"]) or not np.isfinite(values).all():
                        raise ValueError(f"{domain} invalid array: {name} {values.shape}")
                for label, terms in design["contrasts"].items():
                    if not terms.keys() <= names:
                        raise ValueError(f"{domain} contrast {label} has missing terms")
                ordinary = profile["bank_cases"] if domain == "simbank" else profile["test_workloads"]
                for name in required:
                    if len(arrays[name]) != ordinary:
                        raise ValueError(f"{domain} arm {name} has {len(arrays[name])}, expected {ordinary}")
            if domain == "simbank":
                if design["validation_range"][1] > design["case_range"][0]:
                    raise ValueError("SimBank validation and test ranges overlap")
                logging = read(folder / "logging" / "design.json")
                if logging["forecast_family"] != "Regression":
                    raise ValueError("SimBank logging stress did not use Regression")
                a, b = design["case_range"], logging["case_range"]
                if max(a[0], b[0]) < min(a[1], b[1]):
                    raise ValueError("SimBank direct and logging test ranges overlap")
            checks[domain].append({"seed": seed, "arms": len(names), "contrasts": len(design["contrasts"])})
    results = read(root / "results.json")
    if results.get("failures") or set(results.get("domains", {})) != set(EXPECTED):
        raise ValueError("Aggregate is missing a domain or contains failures")
    report = {"status": "pass", "mode": mode, "seeds": expected_seeds, "checks": checks,
              "manifest_sha256": sha256((root / "manifest.json").read_bytes()),
              "results_sha256": sha256((root / "results.json").read_bytes())}
    save(root / "audit.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.run)
    print(json.dumps(result, indent=2))
