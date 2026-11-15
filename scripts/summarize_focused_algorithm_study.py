"""Verify and summarize the focused Regression plus RL study."""

import argparse
import json
from pathlib import Path

import numpy as np


FAMILIES = ("Regression", "RandomForest", "CatBoost", "XGBoost")
ALGORITHMS = ("TabularQ", "FittedQ", "CQL_0.1", "CQL_0.5", "PPO")
CELL = "load_1.2_effect_0.25"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(path, value):
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def discovery_winners(root):
    aggregate = read(root / "aggregate.json")
    winners = {}
    for domain, metric in (("insurance", "queue"), ("bpisim", "queue"),
                           ("manufacturing", "tail_queue")):
        policies = aggregate[domain][CELL]["policies"]
        values = {family: policies[f"{family}_ML_slot"][metric]["mean"] for family in FAMILIES}
        rl_values = {name: policies[name][metric]["mean"] for name in ALGORITHMS}
        winners[domain] = {"metric": metric, "direction": "lower", "ml_values": values,
                           "ml_winner": min(values, key=values.get), "rl_values": rl_values,
                           "rl_winner": min(rl_values, key=rl_values.get)}
    policies = aggregate["simbank"]["negotiation"]["policies"]
    values = {family: policies[f"{family}_DP_500"]["profit"]["mean"] for family in FAMILIES}
    rl_values = {name: policies[name]["profit"]["mean"] for name in ALGORITHMS}
    winners["simbank"] = {"metric": "profit", "direction": "higher", "ml_values": values,
                           "ml_winner": max(values, key=values.get), "rl_values": rl_values,
                           "rl_winner": max(rl_values, key=rl_values.get)}
    return winners


def hierarchical_interval(panels, rng, repetitions=2500, case_level=False):
    if case_level:
        within = np.asarray([rng.multivariate_normal(panel.mean(axis=0),
                            np.cov(panel, rowvar=False) / len(panel), size=repetitions, method="svd")
                             for panel in panels])
    else:
        within = np.asarray([panel[rng.integers(len(panel), size=(repetitions, len(panel)))].mean(axis=1)
                             for panel in panels])
    sampled_seeds = rng.integers(len(panels), size=(repetitions, len(panels)))
    sampled_draws = rng.integers(repetitions, size=sampled_seeds.shape)
    combined = within[sampled_seeds, sampled_draws].mean(axis=1)
    return np.quantile(combined, [0.025, 0.95, 0.975], axis=0)


def validation_choice(report, methods, domain, settings, selection):
    rows = report["validation"]
    reference = rows[report["matched_ml"]]
    fifo = rows["FIFO"] if domain != "simbank" else None
    available = [name for name in methods if name in rows]
    if domain == "simbank":
        limit = selection["refusal_increase_limit_per_case"]
        cancel = settings["cancellation_tolerance_vs_ml"]
        bank = rows["Bank"]
        feasible = [name for name in available
                    if rows[name]["refusals"] <= reference["refusals"] + limit
                    and rows[name]["cancelled"] <= reference["cancelled"] + cancel
                    and rows[name]["cancelled"] <= bank["cancelled"] + settings["cancellation_tolerance_vs_bank"]]
        pool = feasible or available
        return max(pool, key=lambda name: rows[name]["profit"]), bool(feasible)
    metric = selection["primary"][domain]["metric"]
    limits = settings["limits"]
    repeat = selection["repeat_increase_limit_per_case"]
    feasible = [name for name in available
                if rows[name]["extra_attempts"] <= reference["extra_attempts"] + repeat
                and rows[name]["elapsed"] <= limits["elapsed_ratio"] * reference["elapsed"]
                and rows[name]["work"] <= limits["work_ratio"] * reference["work"]
                and rows[name]["unresolved"] <= reference["unresolved"] + limits["resolution_loss"]
                and rows[name]["specialist_work"] <= limits["specialist_share"] * fifo["work"]]
    pool = feasible or available
    return min(pool, key=lambda name: rows[name][metric]), bool(feasible)


def contrast(paths, reports, methods, metric, direction, simbank, rng, settings, candidate_weights=None):
    fields = reports[0]["fields"]
    files = [np.load(path / ("heldout.npz" if simbank else f"{CELL}.npz")) for path in paths]
    try:
        output = {}
        for name, actual in methods.items():
            panels = []
            weights = ([1.] * len(reports) if candidate_weights is None else candidate_weights.get(name, [1.] * len(reports)))
            for arrays, report, candidate, weight in zip(files, reports, actual, weights, strict=True):
                reference = arrays[report["matched_ml"]]
                chosen = reference + weight * (arrays[candidate] - reference)
                delta = chosen - reference
                if simbank:
                    bank = arrays["Bank"]
                    margins = np.column_stack((
                        chosen[:, fields.index("cancelled")] - reference[:, fields.index("cancelled")]
                        - settings["cancellation_tolerance_vs_ml"],
                        chosen[:, fields.index("cancelled")] - bank[:, fields.index("cancelled")]
                        - settings["cancellation_tolerance_vs_bank"]))
                else:
                    fifo = arrays["FIFO"]
                    limits = settings["limits"]
                    margins = np.column_stack((
                        chosen[:, fields.index("elapsed")] - limits["elapsed_ratio"] * reference[:, fields.index("elapsed")],
                        chosen[:, fields.index("work")] - limits["work_ratio"] * reference[:, fields.index("work")],
                        chosen[:, fields.index("unresolved")] - reference[:, fields.index("unresolved")] - limits["resolution_loss"],
                        chosen[:, fields.index("specialist_work")] - limits["specialist_share"] * fifo[:, fields.index("work")]))
                panels.append(np.column_stack((delta, margins)))
            interval = hierarchical_interval(panels, rng, case_level=simbank)
            delta = np.asarray([panel.mean(axis=0) for panel in panels]).mean(axis=0)[:len(fields)]
            signed = -1 if direction == "lower" else 1
            index = fields.index(metric)
            gain = signed * delta[index]
            gain_interval = sorted((signed * interval[0, index], signed * interval[2, index]))
            reference = np.mean([arrays[report["matched_ml"]][:, index].mean()
                                 for arrays, report in zip(files, reports, strict=True)])
            margin_names = (["cancellation_vs_ml", "cancellation_vs_bank"] if simbank else
                            ["elapsed_vs_ml", "work_vs_ml", "resolution_vs_ml", "specialist_capacity"])
            margin_upper = interval[1, len(fields):]
            row = {"methods_by_seed": actual, "reference_by_seed": [r["matched_ml"] for r in reports],
                   "difference": dict(zip(fields, delta.tolist(), strict=True)),
                   "paired_ci95": dict(zip(fields, interval[[0, 2], :len(fields)].T.tolist(), strict=True)),
                   "operating_margin_upper95": dict(zip(margin_names, margin_upper.tolist(), strict=True)),
                   "operating_pass": bool((margin_upper <= 0).all()),
                   "primary_gain": float(gain), "primary_gain_ci95": [float(x) for x in gain_interval],
                   "primary_gain_fraction": float(gain / max(abs(reference), 1e-12))}
            output[name] = row
        return output
    finally:
        for file in files:
            file.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/focused_algorithm_study.json"))
    args = parser.parse_args()
    config = read(args.config)
    manifest = read(args.run / "manifest.json")
    if manifest["config"] != config:
        raise ValueError("Run manifest does not match the focused study config")
    indices = list(range(1 if manifest["mode"] == "smoke" else config["replicates"]))
    selection = config["selection"]
    winners = discovery_winners(Path(selection["discovery_run"]))
    if any(row["ml_winner"] != selection["fixed_forecaster"] for row in winners.values()):
        raise ValueError(f"Discovery winner mismatch: {winners}")
    for domain, locked in selection["primary"].items():
        if winners[domain]["rl_winner"] != locked["algorithm"]:
            raise ValueError(f"Locked RL winner mismatch for {domain}: {winners[domain]}")
    rng = np.random.default_rng(np.random.SeedSequence([config["master_seed"], 919]))
    domains = {}
    for domain in ("insurance", "bpisim", "manufacturing", "simbank"):
        paths = [args.run / domain / f"seed_{i}" for i in indices]
        reports = [read(path / "results.json") for path in paths]
        expected = "Regression_DP_500" if domain == "simbank" else "Regression_ML_slot"
        if any(report["forecast_family"] != "Regression" or report["matched_ml"] != expected for report in reports):
            raise ValueError(f"{domain} did not use the frozen Regression control")
        algorithms = selection["algorithms"] if domain == "simbank" else selection["routing_secondary"]
        locked = selection["primary"][domain]
        validation = [validation_choice(report, algorithms, domain,
                      config["simbank" if domain == "simbank" else "routing"], selection) for report in reports]
        methods = {name: [name] * len(reports) for name in algorithms}
        methods["ValidationSelected"] = [row[0] for row in validation]
        settings = config["simbank" if domain == "simbank" else "routing"]
        results = contrast(paths, reports, methods, locked["metric"], locked["direction"],
                           domain == "simbank", rng, settings)
        primary = results[locked["algorithm"]]
        primary["minimum_gain_pass"] = primary["primary_gain_fraction"] >= selection["minimum_primary_gain_fraction"]
        primary["interval_excludes_zero"] = primary["primary_gain_ci95"][0] > 0
        if domain == "simbank":
            refusal_upper = primary["paired_ci95"]["refusals"][1]
            primary["loop_safety_pass"] = refusal_upper <= selection["refusal_increase_limit_per_case"]
        else:
            repeat_upper = primary["paired_ci95"]["extra_attempts"][1]
            primary["loop_safety_pass"] = repeat_upper <= selection["repeat_increase_limit_per_case"]
        primary["passed"] = bool(primary["minimum_gain_pass"] and primary["interval_excludes_zero"]
                                 and primary["loop_safety_pass"] and primary["operating_pass"])
        domains[domain] = {"control": expected, "locked_primary": locked,
                           "validation_selected_by_seed": [row[0] for row in validation],
                           "validation_selection_feasible_by_seed": [row[1] for row in validation],
                           "comparisons": results}
    save(args.run / "focused_results.json", {"status": "smoke_only" if manifest["mode"] == "smoke" else "full",
         "purpose": "Test whether sequential control adds value after freezing the best Table 2 ML family.",
         "discovery_winners": winners, "selection_protocol": selection,
         "seeds": manifest["seeds"][:len(indices)], "domains": domains})
    print(f"Wrote {args.run / 'focused_results.json'}")


if __name__ == "__main__":
    main()
