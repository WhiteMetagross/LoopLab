"""Summarize the focused Table 3 refinement study."""

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.summarize_focused_algorithm_study import CELL, contrast, read, save


def means(paths, reports, candidates, metric, simbank, weights):
    control, candidate = [], []
    for path, report, name, weight in zip(paths, reports, candidates, weights, strict=True):
        file = path / ("heldout.npz" if simbank else f"{CELL}.npz")
        with np.load(file) as arrays:
            index = report["fields"].index(metric)
            reference = arrays[report["matched_ml"]]
            chosen = reference + weight * (arrays[name] - reference)
            control.append(float(reference[:, index].mean()))
            candidate.append(float(chosen[:, index].mean()))
    return {"control_by_seed": control, "candidate_by_seed": candidate,
            "control_mean": float(np.mean(control)), "candidate_mean": float(np.mean(candidate))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/focused_refinement_study.json"))
    args = parser.parse_args()
    config = read(args.config)
    manifest = read(args.run / "manifest.json")
    if manifest["config"] != config:
        raise ValueError("Run manifest does not match the refinement config")
    indices = list(range(1 if manifest["mode"] == "smoke" else config["replicates"]))
    rng = np.random.default_rng(np.random.SeedSequence([config["master_seed"], 1217]))
    output = {}
    for domain in ("bpisim", "manufacturing", "simbank"):
        paths = [args.run / domain / f"seed_{i}" for i in indices]
        reports = [read(path / "results.json") for path in paths]
        spec = config["selection"]["primary"][domain]
        simbank = domain == "simbank"
        if simbank:
            candidates = [report["selected_profit_controller"] for report in reports]
            if any(name is None for name in candidates):
                raise ValueError("SimBank profit controller selection is missing")
        else:
            candidates = [spec["algorithm"]] * len(reports)
        weights = ([1.] * len(reports) if domain != "manufacturing" else
                   [report["episodic_search"]["safe_blend_weight"] for report in reports])
        settings = config["simbank" if simbank else "routing"]
        row = contrast(paths, reports, {"candidate": candidates}, spec["metric"], spec["direction"],
                       simbank, rng, settings, {"candidate": weights})["candidate"]
        loop = "refusals" if simbank else "extra_attempts"
        limit = config["selection"]["refusal_increase_limit_per_case" if simbank else
                                    "repeat_increase_limit_per_case"]
        absolute_target = config["selection"].get("minimum_absolute_gain", {}).get(domain)
        row["minimum_gain_pass"] = (row["primary_gain"] >= absolute_target if absolute_target is not None else
                                    row["primary_gain_fraction"] >= config["selection"]["minimum_primary_gain_fraction"])
        row["interval_excludes_zero"] = row["primary_gain_ci95"][0] > 0
        row["loop_safety_pass"] = row["paired_ci95"][loop][1] <= limit
        queue_limit = config["selection"].get("queue_increase_limit_fraction", {}).get(domain)
        if queue_limit is not None:
            queue_control = []
            for path, report in zip(paths, reports, strict=True):
                with np.load(path / f"{CELL}.npz") as arrays:
                    queue_control.append(float(arrays[report["matched_ml"]][:, report["fields"].index("queue")].mean()))
            row["queue_safety_pass"] = bool(row["paired_ci95"]["queue"][1]
                                            <= queue_limit * np.mean(queue_control))
        else:
            row["queue_safety_pass"] = True
        row["passed"] = bool(row["minimum_gain_pass"] and row["interval_excludes_zero"]
                             and row["loop_safety_pass"] and row["operating_pass"]
                             and row["queue_safety_pass"])
        output[domain] = {"metric": spec["metric"], "control_by_seed": [r["matched_ml"] for r in reports],
                          "candidate_by_seed": candidates,
                          "candidate_weight_by_seed": weights,
                          "validation_fallback_by_seed": ([r["profit_controller_fallback"] for r in reports] if simbank else
                              [r["episodic_search"]["fallback_to_ml"] for r in reports]),
                          "means": means(paths, reports, candidates, spec["metric"], simbank, weights),
                          "comparison": row}
    save(args.run / "refined_results.json", {"status": "smoke_only" if manifest["mode"] == "smoke" else "full",
         "purpose": "Refine BPISim timing, SimFactory repeat safety and SimBank profit stability on fresh streams.",
         "seeds": manifest["seeds"][:len(indices)], "domains": output,
         "simbank_interval": "Paired case level normal approximation within fit, then fit resampling."})
    print(f"Wrote {args.run / 'refined_results.json'}")


if __name__ == "__main__":
    main()
