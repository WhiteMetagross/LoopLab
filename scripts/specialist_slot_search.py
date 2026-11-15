"""Train a slot policy with episodic return and paired evaluation."""

import argparse
from importlib.metadata import version
import json
from pathlib import Path
import platform
import signal
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_limits

from scripts.specialist_slots_pilot import FIELDS, TAIL_FIELDS, assess, baseline, compare, feasible, fit_grids, gaps, ml_actor, summarize
from src.benchmark_models import seeds
from src.domain_loop_data import acquire, chronological_split, load_cases, selected_domains, sha256
from src.domain_loop_simulations import generate, save
from src.domain_loop_simulator import LoopEnvironment, calibrate
from src.domain_loop_training import read_replay, scenarios


METRICS = {"repeat": 0, "elapsed": 1, "queue": 5, "tail_queue": 7}


def residual_actor(context, weights, slots, history=True):
    intercept, queue, pace, early_queue, pressure, retry, queue_retry, *risk_weights = weights

    def act(ep):
        edge, _, _, risk_standard, risk_specialist, _, _, standard, specialist, attempt, _, remaining, unseen, _, _ = context(ep)
        difference = max(-2., min(2., (standard - specialist) / 4.))
        progress = 1 - unseen / ep.size
        surplus = remaining / slots - unseen / ep.size
        repeated = history and attempt > 1
        adjustment = (intercept + queue * difference + pace * surplus + early_queue * difference * progress
                      + pressure * min(2., standard / 4.) + retry * repeated + queue_retry * difference * repeated)
        if risk_weights:
            effect = risk_standard - risk_specialist
            adjustment += effect * (risk_weights[0] + risk_weights[1] * repeated + risk_weights[2] * surplus)
        if len(risk_weights) > 3:
            waited = ep.accrued_wait[ep.state.case] / ep.env.law.mean_service
            adjustment += risk_weights[3] * difference * min(2., waited / 4.)
        return int(edge + adjustment > 0.)

    return act


def queue_margin(candidate, reference, cfg):
    return candidate[:, 5] - (1 + cfg["queue_noninferiority_fraction"]) * reference[:, 5]


def compare_objective(candidate, reference, fifo, cfg, draws, objective):
    report = compare(candidate, reference, fifo, cfg, draws, objective)
    if "queue_noninferiority_fraction" in cfg:
        upper = float(np.quantile(queue_margin(candidate, reference, cfg)[draws].mean(axis=1), .95))
        report.update(queue_margin_upper95=upper, queue_preserved=upper <= 0, passed=report["passed"] and upper <= 0)
    return report


def run(out, config_path, seed_index=0, namespace=371, selection="insurance", prepare_only=False, fit_source=None, law=None):
    started = perf_counter()
    cfg = json.loads(config_path.read_text())
    objective = cfg.get("objective", "queue")
    variant = cfg.get("ablation", "full")
    history, congestion = variant != "no_loop_history", variant != "no_congestion"
    metric = METRICS[objective]
    config = Path("configs/domain_loop_simulations.json")
    sim = json.loads(config.read_text())
    index, domain, spec = next(iter(selected_domains(sim, selection)))
    simulator = spec["name"]
    seed = seeds(sim["master_seed"], seed_index + 1)[-1]
    entropy = [seed, namespace] if domain == "insurance" else [seed, namespace, index]
    streams = np.random.SeedSequence(entropy).spawn(6)
    fit, online, validation, evaluation, search, bootstrap = streams
    rng = np.random.default_rng(search)
    signal.alarm(cfg["stage_cap_seconds"])
    out.mkdir(parents=True)
    if law is None:
        raw = acquire(spec, sim["maximum_download_bytes"])
        cases, _ = load_cases(raw, domain, spec)
        fitting, _, _, _ = chronological_split(cases, sim["fit_fraction"])
        law = calibrate(fitting, spec, sim)
    ready = cfg.get("route_when_ready", False)
    logs = (fit_source or out) / "ml_fit"
    if fit_source is None:
        logs.mkdir()
        generate(law, spec, sim, fit, logs, cfg["load"], cfg["effect"], cfg["ml_fit_workloads"], cfg["cases_per_workload"], .5, ready)
    replay = read_replay(logs / "transitions.jsonl", ready)
    full_grid = fit_grids(replay, cfg, seed) if fit_source is None else np.load(fit_source / "forecast_grid.npy")
    grid = full_grid if history else fit_grids(replay, cfg, seed, history=False)
    np.save(out / "forecast_grid.npy", grid)
    factory = lambda: LoopEnvironment(law, spec, sim, cfg["effect"], scenario=True, route_when_ready=ready)
    supervised_objective = "elapsed" if objective == "elapsed" else "repeat"
    contexts = {f"ML_wait_{weight:g}": baseline(factory(), full_grid, replay, cfg, "queue_aware", supervised_objective, weight)
                for weight in cfg["baseline_wait_weights"]}
    if objective == "tail_queue":
        contexts.update({f"ML_tail_{weight:g}": baseline(factory(), grid, replay, cfg, "queue_aware", objective, weight)
                         for weight in cfg["tail_baseline_wait_weights"]})
    anchor = "ML_tail" if objective == "tail_queue" else "ML_wait"
    anchor_weight = cfg.get("anchor_wait_weight", .04)
    context = contexts[f"{anchor}_{anchor_weight:g}"]
    if not (history and congestion):
        context = baseline(factory(), grid, replay, cfg, "queue_aware",
                           objective if objective == "tail_queue" else supervised_objective, anchor_weight, congestion)
    controls = {"FIFO": lambda _: 0, "Spend_first": lambda _: 1,
                "Shorter_queue": lambda ep: int(ep.state.specialist_wait_estimate < ep.state.standard_wait_estimate),
                **{name: ml_actor(ctx) for name, ctx in contexts.items()}}
    blocks = [scenarios(stream, law, spec, cfg, cfg["workloads_per_search_block"]) for stream in online.spawn(cfg["search_blocks"])]
    references, repeat_floors = [], []
    for block in blocks:
        rows = assess(factory, block, controls, cfg)
        repeat_floor = min(value[:, 0].mean() for value in rows.values())
        repeat_floors.append(repeat_floor)
        if objective == "tail_queue":
            eligible = (value for value in rows.values() if value[:, 0].mean() <= repeat_floor + cfg["repeat_noninferiority"])
            reference = min(eligible, key=lambda values: values[:, metric].mean())
        else:
            reference = min(rows.values(), key=lambda values: (values[:, 1 if objective == "elapsed" else 0].mean(), values[:, 5].mean()))
        references.append(reference.mean(axis=0))
    spread = np.array([.04, .2, .08, .15, .08, .06, .15])
    if objective == "repeat":
        spread = np.r_[spread, .5, .4, .35]
    elif objective in ("elapsed", "tail_queue"):
        spread = np.asarray(cfg["initial_search_spread"])
    center = np.zeros(len(spread))
    repeat_slack = cfg.get("training_repeat_slack")
    fitted, training = {}, []
    for step in range(cfg["search_rounds"]):
        block = step % len(blocks)
        anchors = [center] if repeat_slack is None else [center, np.zeros_like(center)]
        weights = np.vstack((*anchors, rng.normal(center, spread, (cfg["population"] - len(anchors), len(center)))))
        if variant == "static_ml":
            weights[:, 1:] = 0.
        population = {str(i): residual_actor(context, row, cfg["slots"], history) for i, row in enumerate(weights)}
        summaries = assess(factory, blocks[block], population, cfg)
        costs, repeat_excess = [], []
        for row in summaries.values():
            repeats, wait = row[:, [0, 5]].mean(axis=0)
            if repeat_slack is not None:
                repeat_excess.append(max(0., repeats - repeat_floors[block] - repeat_slack))
            if objective == "repeat":
                excess = wait - (1 + cfg["training_queue_margin_fraction"]) * references[block][5]
                costs.append(100 * repeats + cfg["training_queue_penalty"] * max(0., excess))
            elif objective in ("elapsed", "tail_queue"):
                excess_queue = wait - (1 + cfg["training_queue_margin_fraction"]) * references[block][5]
                excess_repeat = repeats - references[block][0] - cfg["repeat_noninferiority"] / 2
                costs.append(row[:, metric].mean() + cfg["training_queue_penalty"] * max(0., excess_queue)
                             + cfg["training_repeat_penalty"] * max(0., excess_repeat))
            else:
                extra = repeats - references[block][0]
                costs.append(wait + 650 * max(0., extra - .002) + 3000 * max(0., extra - cfg["repeat_noninferiority"]))
        ranking = np.argsort(costs) if repeat_slack is None else np.lexsort((costs, repeat_excess))
        elite = ranking[:cfg["elite"]]
        best = int(elite[0])
        name = f"RL_round_{step + 1}"
        fitted[name] = weights[best].tolist()
        training.append({"name": name, "block": block, "return_cost": float(costs[best])})
        if repeat_slack is not None:
            training[-1]["training_repeat_excess"] = float(repeat_excess[best])
        center = .3 * center + .7 * weights[elite].mean(axis=0)
        spread = np.maximum(.007, .4 * spread + .6 * weights[elite].std(axis=0))
    candidates = {name: residual_actor(context, weights, cfg["slots"], history) for name, weights in fitted.items()}
    validation_workloads = scenarios(validation, law, spec, cfg, cfg["validation_workloads"])
    val = assess(factory, validation_workloads, controls | candidates, cfg)
    feasible_controls = [name for name in controls if name != "FIFO" and feasible(val[name], val["FIFO"], cfg["limits"])]
    feasible_ml = [name for name in feasible_controls if name.startswith("ML_")]
    if not feasible_ml:
        save(out / "results.json", {"simulator": simulator, "seed": seed, "settings": cfg, "validation": summarize(val),
                                    "status": "no_feasible_ml", "passed": False})
        print(f"{simulator} queue search: no feasible ML comparator", flush=True)
        signal.alarm(0)
        return
    if objective in ("repeat", "elapsed"):
        rank = lambda name: (val[name][:, metric].mean(), val[name][:, 5].mean())
        reference, learned = min(feasible_controls, key=rank), min(feasible_ml, key=rank)

        def constraint(name):
            return all(queue_margin(val[name], val[key], cfg).mean() <= 0
                       and (objective != "elapsed" or (val[name][:, 0] - val[key][:, 0]).mean() <= cfg["repeat_noninferiority"])
                       for key in (reference, learned))
    else:
        min_repeat = min(val[name][:, 0].mean() for name in feasible_controls)
        near_best = [name for name in feasible_controls if val[name][:, 0].mean() <= min_repeat + cfg["repeat_noninferiority"]]
        reference = min(near_best, key=lambda name: val[name][:, metric].mean())
        near_best_ml = [name for name in near_best if name.startswith("ML_")]
        learned = (min(near_best_ml, key=lambda name: val[name][:, metric].mean()) if near_best_ml else
                   min(feasible_ml, key=lambda name: (val[name][:, 0].mean(), val[name][:, metric].mean())))

        def constraint(name):
            return (val[name][:, 0].mean() <= min_repeat + cfg["repeat_noninferiority"] and
                    (objective != "tail_queue" or all(queue_margin(val[name], val[key], cfg).mean() <= 0
                                                      for key in (reference, learned))))
    selected = min((name for name in candidates if feasible(val[name], val["FIFO"], cfg["limits"], val[reference])
                    and constraint(name)), default=None, key=lambda name: val[name][:, metric].mean())
    if selected and val[selected][:, metric].mean() >= val[reference][:, metric].mean():
        selected = None
    freeze = {"simulator": simulator, "domain": domain, "seed": seed, "seed_index": seed_index, "namespace": namespace, "settings": cfg,
              "objective": objective,
              "selected_rl": selected, "best_ml": learned, "best_control": reference,
              "fit_source": str(fit_source) if fit_source is not None else None,
              "validation": summarize(val), "weights": fitted, "training": training,
              "sha256": {str(p): sha256(p.read_bytes()) for p in (config_path, config, Path(__file__),
                               Path("scripts/specialist_slots_pilot.py"), Path("scripts/audit_routing_rewards.py"),
                               Path("src/domain_loop_simulator.py"), Path("src/domain_loop_data.py"),
                               Path("src/domain_loop_simulations.py"), Path("src/domain_loop_training.py"),
                               Path("src/bpisim_data.py"), Path("src/benchmark_models.py"))}}
    save(out / "freeze.json", freeze)
    if prepare_only:
        signal.alarm(0)
        return
    print(f"{simulator} {objective} search: ML={learned}, strongest={reference}, RL={selected}", flush=True)
    tested = {"FIFO": controls["FIFO"], **{k: controls[k] for k in dict.fromkeys((learned, reference))},
              **{name: candidates[name] for name in dict.fromkeys((selected, next(reversed(fitted)))) if name}}
    final = assess(factory, scenarios(evaluation, law, spec, cfg, cfg["evaluation_workloads"]), tested, cfg)
    assert all(np.allclose(rows[:, FIELDS.index("slots")] * cfg["cases_per_workload"], 0 if name == "FIFO" else cfg["slots"])
               for name, rows in final.items())
    count = cfg["evaluation_workloads"]
    draws = np.random.default_rng(bootstrap).integers(count, size=(cfg["bootstrap_draws"], count))
    comparisons = {name: {ref: compare_objective(final[name], final[ref], final["FIFO"], cfg, draws, objective)
                          for ref in dict.fromkeys((learned, reference))} for name in tested if name in candidates}
    np.savez_compressed(out / "paired_workloads.npz", **final)
    report = freeze | {"evaluation": summarize(final), "comparisons": comparisons,
                       "passed": bool(selected and all(c["passed"] for c in comparisons[selected].values())),
                       "seconds": perf_counter() - started, "device": "CPU", "python": platform.python_version(),
                       "versions": {name: version(name) for name in ("numpy", "catboost", "scikit-learn")}}
    save(out / "results.json", report)
    print(f"{simulator} {objective} search: passed={report['passed']}", flush=True)
    signal.alarm(0)


def confirm(out, config_path, namespace, selection):
    out.mkdir(parents=True)
    for index in range(3):
        run(out / f"seed_{index}", config_path, index, namespace, selection)
    summarize_confirmation(out, namespace)


def summarize_confirmation(out, namespace):
    reports = [json.loads((out / f"seed_{index}" / "results.json").read_text()) for index in range(3)]
    if any(report.get("status") == "no_feasible_ml" for report in reports):
        save(out / "aggregate.json", {"status": "no_feasible_ml", "seeds": [report["seed"] for report in reports], "passed": False})
        return
    cfg = reports[0]["settings"]
    objective = reports[0]["objective"]
    metric = METRICS[objective]
    fields = TAIL_FIELDS if objective == "tail_queue" else FIELDS
    limit = cfg["limits"]
    rng = np.random.default_rng(np.random.SeedSequence([*(report["seed"] for report in reports), namespace]))
    repetitions = cfg["bootstrap_draws"]
    comparisons = {}
    for comparator in ("best_ml", "best_control"):
        panels, summaries = [], []
        for index, report in enumerate(reports):
            with np.load(out / f"seed_{index}" / "paired_workloads.npz") as rows:
                selected = report["selected_rl"] or report["best_control"]
                candidate, reference, fifo = rows[selected], rows[report[comparator]], rows["FIFO"]
            delta = candidate - reference
            margins = np.column_stack((gaps(candidate, fifo, limit), gaps(candidate, reference, limit),
                                       candidate[:, 3] - limit["specialist_share"] * fifo[:, 2]))
            if "queue_noninferiority_fraction" in cfg:
                margins = np.column_stack((margins, queue_margin(candidate, reference, cfg)))
            panels.append(np.column_stack((delta, margins)))
            summaries.append({"seed": report["seed"], "selected_rl": report["selected_rl"], "reference": report[comparator],
                              "ml": summarize({"row": reference})["row"], "candidate": summarize({"row": candidate})["row"],
                              "queue_delta": float(delta[:, 5].mean()), "repeat_delta": float(delta[:, 0].mean()),
                              "elapsed_delta": float(delta[:, 1].mean()), "primary_delta": float(delta[:, metric].mean())})
        within = np.array([panel[rng.integers(len(panel), size=(repetitions, len(panel)))].mean(axis=1) for panel in panels])
        sampled = rng.integers(len(panels), size=(repetitions, len(panels)))
        indices = rng.integers(repetitions, size=sampled.shape)
        combined = within[sampled, indices].mean(axis=1)
        intervals = np.quantile(combined[:, :len(fields)], [.025, .975], axis=0)
        differences = np.array(panels).mean(axis=(0, 1))[:len(fields)]
        margin_upper = np.quantile(combined[:, len(fields):], .95, axis=0)
        checks = {"operating_bounds_pass": bool((margin_upper[:7] <= 0).all()),
                  "all_seeds_selected_and_improved": all(row["selected_rl"] and row["primary_delta"] < 0 for row in summaries)}
        if objective == "repeat":
            checks.update(repeat_minimum_reduction=bool(differences[0] <= -cfg["minimum_repeat_reduction"]),
                          repeat_interval_excludes_zero=bool(intervals[1, 0] < 0), queue_preserved=bool(margin_upper[-1] <= 0))
        elif objective == "elapsed":
            checks.update(elapsed_minimum_reduction=bool(differences[1] <= -cfg["minimum_elapsed_reduction_minutes"]),
                          elapsed_interval_excludes_zero=bool(intervals[1, 1] < 0), queue_preserved=bool(margin_upper[-1] <= 0),
                          repeats_preserved=bool(intervals[1, 0] <= cfg["repeat_noninferiority"]))
        elif objective == "tail_queue":
            reference_tail = np.mean([row["ml"]["tail_queue"] for row in summaries])
            checks.update(tail_queue_minimum_reduction=bool(differences[metric] <= -cfg["minimum_tail_queue_reduction_fraction"] * reference_tail),
                          tail_queue_interval_excludes_zero=bool(intervals[1, metric] < 0), queue_preserved=bool(margin_upper[-1] <= 0),
                          repeats_preserved=bool(intervals[1, 0] <= cfg["repeat_noninferiority"]))
        else:
            reference_queue = np.mean([row["ml"]["queue"] for row in summaries])
            checks.update(queue_5_percent=bool(differences[5] <= -cfg["minimum_queue_reduction_fraction"] * reference_queue),
                          queue_interval_excludes_zero=bool(intervals[1, 5] < 0), repeats_preserved=bool(intervals[1, 0] <= cfg["repeat_noninferiority"]))
        comparisons[comparator] = {"by_seed": summaries, "difference": dict(zip(fields, differences.tolist(), strict=True)),
                                   "queue_difference_seed_sd": float(np.std([row["queue_delta"] for row in summaries], ddof=1)),
                                   "primary_difference_seed_sd": float(np.std([row["primary_delta"] for row in summaries], ddof=1)),
                                   "paired_hierarchical_ci95": dict(zip(fields, intervals.T.tolist(), strict=True)),
                                   "operating_margin_upper95": margin_upper.tolist(), "checks": checks, "passed": all(checks.values())}
    report = {"simulator": reports[0]["simulator"], "objective": objective, "namespace": namespace, "settings": cfg, "seeds": [entry["seed"] for entry in reports],
              "comparisons": comparisons, "passed": all(value["passed"] for value in comparisons.values())}
    save(out / "aggregate.json", report)
    print(f"{report['simulator']} three-seed confirmation: passed={report['passed']}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/specialist_slot_search.json"))
    parser.add_argument("--domain", type=str.lower, default="simclaims",
                        choices=("simclaims", "simfactory", "bpisim", "insurance", "manufacturing"))
    parser.add_argument("--confirm", action="store_true")
    parser.add_argument("--namespace", type=int)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        if args.confirm:
            confirm(args.out, args.config, namespace=613 if args.namespace is None else args.namespace, selection=args.domain)
        else:
            run(args.out, args.config, namespace=371 if args.namespace is None else args.namespace, selection=args.domain)
