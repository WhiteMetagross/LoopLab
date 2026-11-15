"""Evaluate validation-tuned controller mixtures on fresh repair workloads."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.optimize import linprog
from threadpoolctl import threadpool_limits

from scripts.run_paper_suite import intervals
from scripts.specialist_slot_search import residual_actor
from scripts.specialist_slots_pilot import TAIL_FIELDS, assess, baseline, gaps, ml_actor, summarize
from src.benchmark_models import seeds
from src.domain_loop_data import acquire, chronological_split, load_cases, sha256
from src.domain_loop_simulations import save
from src.domain_loop_simulator import LoopEnvironment, calibrate
from src.domain_loop_training import read_replay, scenarios


REPEAT_MARGIN = .005
QUEUE_RATIO = 1.05
TAIL_TARGET = .05
FRESH_STREAM = 2719


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def validation_caps(validation, reference, limits):
    ref, fifo = validation[reference], validation["FIFO"]
    return {"extra_attempts": ref["extra_attempts"] + REPEAT_MARGIN, "queue": QUEUE_RATIO * ref["queue"],
            "elapsed": limits["elapsed_ratio"] * min(ref["elapsed"], fifo["elapsed"]),
            "work": limits["work_ratio"] * min(ref["work"], fifo["work"]),
            "unresolved": min(ref["unresolved"], fifo["unresolved"]) + limits["resolution_loss"],
            "specialist_work": limits["specialist_share"] * fifo["work"]}


def fit_mix(validation, names, reference, limits):
    caps = validation_caps(validation, reference, limits)
    matrix = np.asarray([[validation[name][metric] for metric in caps] for name in names])
    cost = [validation[name]["tail_queue"] for name in names]
    result = linprog(cost, A_ub=matrix.T, b_ub=list(caps.values()), A_eq=np.ones((1, len(names))), b_eq=[1],
                     bounds=(0, 1), method="highs")
    if not result.success:
        raise ValueError(f"No feasible validation mixture: {result.message}")
    return {name: float(weight) for name, weight in zip(names, result.x, strict=True) if weight > 1e-9}


def weighted_rows(weights, rows):
    np.testing.assert_allclose(sum(weights.values()), 1.)
    return sum(weight * rows[name] for name, weight in weights.items())


def make_controls(frozen, grid, replay, factory):
    settings = frozen["settings"]
    contexts = {}
    controls = {"FIFO": lambda _: 0, "Spend_first": lambda _: 1,
                "Shorter_queue": lambda ep: int(ep.state.specialist_wait_estimate < ep.state.standard_wait_estimate)}
    for name in frozen["validation"]:
        if name.startswith("ML_"):
            _, variant, weight = name.split("_")
            contexts[name] = baseline(factory(), grid, replay, settings, "queue_aware",
                                      "tail_queue" if variant == "tail" else "repeat", float(weight))
            controls[name] = ml_actor(contexts[name])
    anchor = contexts[f"ML_tail_{settings.get('anchor_wait_weight', .04):g}"]
    controls.update({name: residual_actor(anchor, weights, settings["slots"]) for name, weights in frozen["weights"].items()})
    return controls


def select_mixtures(frozen, names):
    reference, validation, limits = frozen["best_control"], frozen["validation"], frozen["settings"]["limits"]
    names = [name for name in names if name != "FIFO"]
    ml = fit_mix(validation, [name for name in names if not name.startswith("RL_")], reference, limits)
    hybrid = fit_mix(validation, names, reference, limits)

    def expected(weights):
        return {metric: sum(weight * validation[name][metric] for name, weight in weights.items()) for metric in TAIL_FIELDS}

    return {"seed": frozen["seed"], "reference": reference, "ml_mix": ml, "hybrid_mix": hybrid,
            "validation_ml": expected(ml), "validation_hybrid": expected(hybrid)}


def summarize_comparison(panels, limits, seed):
    margins, deltas, seeds_ = [], [], []
    for row in panels:
        candidate, baseline, fifo = (row[key] for key in ("hybrid_mix", "ml_mix", "FIFO"))
        delta = candidate - baseline
        deltas.append(delta)
        margins.append(np.column_stack((gaps(candidate, fifo, limits), gaps(candidate, baseline, limits),
                       candidate[:, 3] - limits["specialist_share"] * fifo[:, 2],
                       candidate[:, 5] - QUEUE_RATIO * baseline[:, 5])))
        seeds_.append({"ml": summarize({"row": baseline})["row"], "hybrid": summarize({"row": candidate})["row"],
                       "tail_difference": float(delta[:, 7].mean())})
    arrays = [np.column_stack((delta, margin)) for delta, margin in zip(deltas, margins, strict=True)]
    bands = intervals(arrays, np.random.default_rng(seed), 4000, case_level=False)
    mean = np.concatenate(deltas).mean(axis=0)
    baseline_tail = np.mean([row["ml"]["tail_queue"] for row in seeds_])
    n = len(TAIL_FIELDS)
    checks = {"tail_minimum": bool(mean[7] <= -TAIL_TARGET * baseline_tail), "tail_interval": bool(bands[2, 7] < 0),
              "repeat_preserved": bool(bands[2, 0] <= REPEAT_MARGIN), "queue_preserved": bool(bands[1, -1] <= 0),
              "operating_limits": bool((bands[1, n:-1] <= 0).all()),
              "tail_improved_every_fit": all(row["tail_difference"] < 0 for row in seeds_)}
    return {"checks": checks, "passed": all(checks.values()), "by_seed": seeds_,
            "delta": dict(zip(TAIL_FIELDS, mean.tolist(), strict=True)),
            "tail_gain_fraction": float(-mean[7] / baseline_tail),
            "paired_ci95": dict(zip(TAIL_FIELDS, bands[[0, 2], :n].T.tolist(), strict=True)),
            "constraint_upper95": bands[1, n:].tolist()}


def run(out, parent):
    start = perf_counter()
    domain_config = Path("configs/domain_loop_simulations.json")
    sim = read(domain_config)
    spec = sim["domains"]["manufacturing"]
    master = sim["master_seed"]
    fitting_seeds = seeds(master, 3)
    raw = acquire(spec, sim["maximum_download_bytes"])
    cases, _ = load_cases(raw, "manufacturing", spec)
    fit, _, _, _ = chronological_split(cases, sim["fit_fraction"])
    law = calibrate(fit, spec, sim)
    out.mkdir(parents=True)
    inputs = {str(path): sha256(path.read_bytes()) for path in (Path(__file__), domain_config,
              Path("scripts/specialist_slots_pilot.py"), Path("scripts/specialist_slot_search.py"), Path("scripts/run_paper_suite.py"))}
    results, freezes = [], []
    for index, seed in enumerate(fitting_seeds):
        source = parent / f"seed_{index}"
        grid_file, freeze_file = source / "forecast_grid.npy", source / "freeze.json"
        replay_file = source / "ml_fit" / "transitions.jsonl"
        inputs.update({str(path): sha256(path.read_bytes()) for path in (grid_file, freeze_file, replay_file)})
        frozen = read(freeze_file)
        assert frozen["seed"] == seed and frozen["objective"] == "tail_queue"
        cfg = frozen["settings"]
        factory = lambda: LoopEnvironment(law, spec, sim, cfg["effect"], scenario=True)
        controls = make_controls(frozen, np.load(grid_file), read_replay(replay_file), factory)
        reference = frozen["best_control"]
        old_stream = np.random.SeedSequence([seed, frozen["namespace"], 1]).spawn(6)[3]
        checked = {name: controls[name] for name in (reference, next(reversed(frozen["weights"])))}
        old = assess(factory, scenarios(old_stream, law, spec, cfg, 2), checked, cfg)
        with np.load(source / "paired_workloads.npz") as saved:
            for name, rows in old.items():
                np.testing.assert_allclose(rows, saved[name][:2], atol=1e-8)
        choice = select_mixtures(frozen, controls)
        ml, hybrid = choice["ml_mix"], choice["hybrid_mix"]
        freezes.append(choice)
        save(out / f"freeze_{index}.json", choice)
        keep = dict.fromkeys(("FIFO", reference, *ml, *hybrid))
        stream = np.random.SeedSequence([seed, FRESH_STREAM, 1])
        rows = assess(factory, scenarios(stream, law, spec, cfg, cfg["evaluation_workloads"]),
                      {name: controls[name] for name in keep}, cfg)
        mixtures = {"ml_mix": weighted_rows(ml, rows), "hybrid_mix": weighted_rows(hybrid, rows),
                    "FIFO": rows["FIFO"], "original_reference": rows[reference]}
        for key in ("ml_mix", "hybrid_mix"):
            np.testing.assert_allclose(mixtures[key][:, 6] * cfg["cases_per_workload"], cfg["slots"])
        np.savez_compressed(out / f"paired_{index}.npz", **mixtures)
        results.append(mixtures)
        print(f"Fit {index}: tail ML {mixtures['ml_mix'][:, 7].mean():.2f}, hybrid {mixtures['hybrid_mix'][:, 7].mean():.2f}", flush=True)
    report = summarize_comparison(results, cfg["limits"], master)
    anchor_report = summarize_comparison([row | {"ml_mix": row["original_reference"]} for row in results], cfg["limits"], master + 1)
    save(out / "results.json", {"comparison": report, "original_reference_comparison": anchor_report,
                                "passed": report["passed"] and anchor_report["passed"], "frozen_mixtures": freezes,
                                "source_sha256": inputs, "stream": FRESH_STREAM,
                                "limits": {"extra_repeats_per_case": REPEAT_MARGIN, "mean_queue_ratio": QUEUE_RATIO, "tail_minimum": TAIL_TARGET},
                                "seconds": perf_counter() - start})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--parent", type=Path, default=Path("runs/reproduction/full/common/simfactory_tail"))
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run(args.out, args.parent)
