"""Create a compact derived summary for the Regression ablation suite."""

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.summarize_focused_algorithm_study import hierarchical_interval
from src.domain_loop_simulations import save


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def effect(row, metric, lower=True):
    mean = row["mean"][metric]
    lo, hi = row["paired_ci95"][metric]
    if lower:
        return {"gain": -mean, "gain_ci95": [-hi, -lo]}
    return {"gain": mean, "gain_ci95": [lo, hi]}


def main(root):
    audit, results = read(root / "audit.json"), read(root / "results.json")
    if audit["status"] != "pass" or results["failures"]:
        raise ValueError("Ablation run has not passed its integrity audit")
    insurance = results["domains"]["insurance"]
    bpi = results["domains"]["bpisim"]
    factory = results["domains"]["manufacturing"]
    bank = results["domains"]["simbank"]
    output = {"status": audit["mode"], "seeds": results["seeds"], "runtime_seconds": results["seconds"]}
    control = insurance["policies"]["full_ML"]["mean"]["queue"]
    output["insurance"] = {
        "full_vs_ml": effect(insurance["comparisons"]["full_minus_supervised_ML"], "queue"),
        "full_vs_ml_fraction": -insurance["comparisons"]["full_minus_supervised_ML"]["mean"]["queue"] / control,
        "full_vs_equal_static": effect(insurance["comparisons"]["full_minus_equal_static"], "queue"),
        "full_vs_no_congestion": effect(insurance["comparisons"]["full_minus_no_congestion"], "queue"),
        "full_vs_no_forecast": effect(insurance["comparisons"]["full_minus_no_forecast"], "queue"),
        "full_vs_no_loop_state": effect(insurance["comparisons"]["full_minus_no_loop_state"], "queue"),
        "full_vs_first_only": effect(insurance["comparisons"]["full_minus_first_only"], "queue"),
    }
    marker = bpi["policies"]["marker_ML"]["mean"]
    output["bpisim"] = {
        "selected_vs_ml": effect(bpi["comparisons"]["marker_selected_minus_ML"], "elapsed"),
        "forced_vs_ml": effect(bpi["comparisons"]["marker_forced_minus_ML"], "elapsed"),
        "reviewer_share": (marker["queue"] + marker["work"]) / marker["elapsed"],
        "fixed_delay_share": (marker["pre_service"] + marker["residual"]) / marker["elapsed"],
        "marker_queue": marker["queue"],
        "ready_queue": bpi["policies"]["ready_ML"]["mean"]["queue"],
    }
    fields = factory["fields"]
    rng = np.random.default_rng(np.random.SeedSequence([17, 991]))
    frontier = {}
    for arm in ("raw", "mean_blend", "safe_050", "safe_075", "safe_100"):
        panels = []
        for index in range(len(results["seeds"])):
            with np.load(root / "manufacturing" / f"seed_{index}" / "paired.npz") as arrays:
                panels.append(arrays[arm] - arrays["ML"])
        mean = np.asarray([panel.mean(axis=0) for panel in panels]).mean(axis=0)
        interval = hierarchical_interval(panels, rng, 2500, case_level=False)
        tail, repeat = fields.index("tail_queue"), fields.index("extra_attempts")
        frontier[arm] = {"tail_gain": float(-mean[tail]),
                         "tail_gain_ci95": [float(-interval[2, tail]), float(-interval[0, tail])],
                         "tail_gain_fraction": float(-mean[tail] / factory["policies"]["ML"]["mean"]["tail_queue"]),
                         "extra_attempts_per_100": float(100 * mean[repeat]),
                         "extra_attempts_per_100_ci95": [float(100 * interval[0, repeat]),
                                                          float(100 * interval[2, repeat])]}
    output["manufacturing"] = {"frontier": frontier,
        "safe_vs_no_accrued_wait": effect(factory["comparisons"]["safe_075_minus_no_accrued_wait"], "tail_queue"),
        "safe_vs_mean_queue_objective": effect(factory["comparisons"]["safe_075_minus_mean_queue_objective"], "tail_queue"),
        "burst_safe_vs_ml": effect(factory["comparisons"]["burst_safe_075_minus_ML"], "tail_queue")}
    output["simbank"] = {
        "both_vs_profit": {"profit": effect(bank["comparisons"]["both_minus_profit"], "profit", False),
                           "refusals_avoided_per_100": effect(bank["comparisons"]["both_minus_profit"], "refusals")},
        "ensemble_vs_single": effect(bank["comparisons"]["ensemble_minus_single"], "profit", False),
        "constrained_vs_fixed_cql": effect(bank["comparisons"]["constrained_minus_fixed_CQL"], "profit", False),
        "full_vs_history_only": effect(bank["comparisons"]["full_CQL_minus_history"], "profit", False),
        "biased_vs_random_logging": effect(bank["comparisons"]["logging_CQL_biased_minus_random"], "profit", False),
        "selector_by_seed": {seed: design["selector"] for seed, design in bank["design_by_seed"].items()},
    }
    output["simbank"]["both_vs_profit"]["refusals_avoided_per_100"] = {
        key: 100 * value if key == "gain" else [100 * x for x in value]
        for key, value in output["simbank"]["both_vs_profit"]["refusals_avoided_per_100"].items()}
    save(root / "derived_summary.json", output)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(main(args.run), indent=2))
