"""Regression focused routing ablations on fresh paired workloads."""

from pathlib import Path

import joblib
import numpy as np

from scripts.audit_routing_rewards import feasible
from scripts.routing_loop_followup import cluster_arrivals
from scripts.specialist_slot_search import residual_actor
from scripts.specialist_slots_pilot import baseline, ml_actor, upper_tail_mean
from src.domain_loop_data import acquire, chronological_split, load_cases, selected_domains, sha256
from src.domain_loop_simulations import metrics, save
from src.domain_loop_simulator import LoopEnvironment, calibrate
from src.domain_loop_training import read_replay, scenarios
from src.paper_routing import FIELDS as BASE_FIELDS, Routing


FIELDS = (*BASE_FIELDS, "pre_service", "residual")
METRIC = {"elapsed": 1, "queue": 5, "tail_queue": 12}


def repeat_upper(candidate, reference, z=1.645):
    delta = candidate[:, 0] - reference[:, 0]
    return float(delta.mean() + z * delta.std(ddof=1) / np.sqrt(len(delta)))


def mixture_weight(repeat_delta, limit, safety=1.0):
    return 1.0 if repeat_delta <= 0 else min(1.0, safety * limit / repeat_delta)


def weighted_rows(reference, candidate, weight):
    return reference + weight * (candidate - reference)


def detailed_metrics(factory, workloads, policies, n):
    result = {}
    for name, policy in policies.items():
        rows = []
        for workload in workloads:
            env = factory(workload)
            while env.state is not None:
                env.step(policy(env.ep))
            cases = env.ep.env.case_rows()
            raw = env.ep.result()
            summary = metrics(cases)
            pre_service = sum(row.service_ready - row.ready for row in env.ep.env.records)
            residual = sum(row.end - row.service_end for row in env.ep.env.records)
            rows.append([*(raw / n), summary["repeat_rate"], summary["p90_elapsed_minutes"],
                         summary["negative_exit_rate"], summary["unobserved_exit_rate"],
                         summary["cap_rate"], upper_tail_mean([row["queue_wait"] for row in cases], .1),
                         pre_service / n, residual / n])
        result[name] = np.asarray(rows)
    return result


def masked_actor(context, weights, slots, history=True, mask=None):
    values = np.asarray(weights, dtype=float).copy()
    if mask is not None:
        values *= np.asarray(mask, dtype=float)
    return residual_actor(context, values, slots, history)


def fit_search(factory, context, ml_policy, train_workloads, validation_workloads, settings,
               search_cfg, objective, seed, history=True, mask=None):
    metric = METRIC[objective]
    blocks = np.array_split(np.asarray(train_workloads, dtype=object), search_cfg["blocks"])
    block_reference = [detailed_metrics(factory, list(block), {"ML": ml_policy}, settings["cases_per_workload"])["ML"]
                       for block in blocks]
    rng = np.random.default_rng(seed)
    center = np.zeros(len(search_cfg["initial_spread"]), dtype=float)
    spread = np.asarray(search_cfg["initial_spread"], dtype=float)
    candidates, training = {}, []
    for step in range(search_cfg["rounds"]):
        block_index = step % len(blocks)
        weights = np.vstack((center, rng.normal(center, spread,
                            size=(search_cfg["population"] - 1, len(center)))))
        actors = {str(i): masked_actor(context, row, settings["slots"], history, mask)
                  for i, row in enumerate(weights)}
        scored = detailed_metrics(factory, list(blocks[block_index]), actors, settings["cases_per_workload"])
        costs = []
        for values in scored.values():
            excess = max(0.0, values[:, 0].mean() - block_reference[block_index][:, 0].mean()
                         - search_cfg["repeat_tolerance"])
            costs.append(values[:, metric].mean() + search_cfg["repeat_penalty"] * excess)
        elite = np.argsort(costs)[:search_cfg["elite"]]
        best = int(elite[0])
        label = f"round_{step + 1}"
        candidates[label] = weights[best] * (np.asarray(mask) if mask is not None else 1.0)
        training.append({"round": step + 1, "cost": float(costs[best]),
                         "weights": candidates[label].tolist()})
        center = .3 * center + .7 * weights[elite].mean(axis=0)
        spread = np.maximum(search_cfg["minimum_spread"],
                            .4 * spread + .6 * weights[elite].std(axis=0))
    actors = {name: masked_actor(context, row, settings["slots"], history) for name, row in candidates.items()}
    validation = detailed_metrics(factory, validation_workloads, actors, settings["cases_per_workload"])
    controls = detailed_metrics(factory, validation_workloads,
                                {"ML": ml_policy, "FIFO": lambda _env: 0}, settings["cases_per_workload"])
    reference, fifo = controls["ML"], controls["FIFO"]
    valid_operating = [name for name, values in validation.items()
                       if feasible(values[:, :13], fifo[:, :13], settings["limits"], reference[:, :13])]
    forced = min(valid_operating or validation, key=lambda name: validation[name][:, metric].mean())
    allowed = [name for name in valid_operating
               if repeat_upper(validation[name], reference, search_cfg["repeat_confidence_z"])
                  <= search_cfg["repeat_tolerance"]
               and validation[name][:, metric].mean()
                  <= (1 - search_cfg["minimum_validation_gain_fraction"]) * reference[:, metric].mean()]
    selected = min(allowed, default=None, key=lambda name: validation[name][:, metric].mean())
    return {"actors": actors, "validation": validation, "reference": reference, "fifo": fifo, "selected": selected,
            "forced": forced, "training": training}


def choose_blend(search, settings, search_cfg, safety, use_upper=True):
    metric = METRIC["tail_queue"]
    options = []
    for name, values in search["validation"].items():
        delta = (repeat_upper(values, search["reference"], search_cfg["repeat_confidence_z"])
                 if use_upper else float((values[:, 0] - search["reference"][:, 0]).mean()))
        weight = mixture_weight(delta, search_cfg["repeat_tolerance"], safety)
        blended = weighted_rows(search["reference"], values, weight)
        if (weight >= search_cfg["minimum_weight"]
            and feasible(blended[:, :13], search["fifo"][:, :13], settings["limits"], search["reference"][:, :13])):
            options.append((name, weight, blended))
    return min(options, default=None, key=lambda row: row[2][:, metric].mean())


def load_replay(source):
    rows = []
    for path in sorted((source / "ml_fit").glob("*/transitions.jsonl")):
        rows.extend(read_replay(path))
    if not rows:
        raise ValueError(f"No fitting replay found in {source}")
    return rows


def run(folder, index, seed, domain, config, profile, progress):
    sim = joblib.load(config["domain_config"]) if str(config["domain_config"]).endswith(".joblib") else None
    if sim is None:
        import json
        sim = json.loads(Path(config["domain_config"]).read_text())
    _, _, spec = next(iter(selected_domains(sim, domain)))
    raw = acquire(spec, sim["maximum_download_bytes"])
    cases, provenance = load_cases(raw, domain, spec)
    fit, historical, boundary, excluded = chronological_split(cases, sim["fit_fraction"])
    law = calibrate(fit, spec, sim)
    source = Path(config["sources"][domain]) / f"seed_{index}"
    frozen = __import__("json").loads((source / "freeze.json").read_text())
    if frozen["forecast_family"] != "Regression":
        raise ValueError("Ablation source must use the Regression forecast family")
    grids, _ = joblib.load(source / "predictors.joblib")
    grid = grids["Regression"]
    replay = load_replay(source)
    settings = config["routing"] | profile["routing"]
    search_cfg = settings["episodic_search"] | profile["episodic_search"]
    search_cfg["repeat_penalty"] = search_cfg.get("repeat_penalty_by_domain", {}).get(
        domain, search_cfg["repeat_penalty"])
    n = settings["cases_per_workload"]
    streams = np.random.SeedSequence([seed, profile["namespace"], {"insurance": 1, "bpisim": 2, "manufacturing": 3}[domain]]).spawn(3)
    training = scenarios(streams[0], law, spec, settings | {"load": settings["primary_load"]}, profile["training_workloads"])
    validation = scenarios(streams[1], law, spec, settings | {"load": settings["primary_load"]}, profile["validation_workloads"])
    testing = scenarios(streams[2], law, spec, settings | {"load": settings["primary_load"]}, profile["test_workloads"])

    raw_env = LoopEnvironment(law, spec, sim, settings["effect"], scenario=True)
    reward_weights = np.asarray(frozen["reward_weights"])

    def make(grid_value=grid, ready=False, first_only=False, arrivals=None):
        def factory(workload):
            item = workload if arrivals is None else (arrivals(workload[0]), workload[1])
            return Routing(LoopEnvironment(law, spec, sim, settings["effect"], scenario=True,
                                           route_when_ready=ready), item, settings["slots"],
                           grid_value, reward_weights, first_only=first_only)
        return factory

    rows, contrasts, design = {}, {}, {"domain": domain, "forecast_family": "Regression"}
    objective = {"insurance": "queue", "bpisim": "elapsed", "manufacturing": "tail_queue"}[domain]

    if domain == "insurance":
        variants = {
            "full": (grid, True, None, False),
            "equal_static": (grid, True, [1] + [0] * (len(search_cfg["initial_spread"]) - 1), False),
            "no_congestion": (grid, True, None, False),
            "no_forecast": (np.broadcast_to(grid.mean(axis=(0, 1, 2)), grid.shape).copy(), True, None, False),
            "no_loop_state": (np.broadcast_to(grid.mean(axis=0, keepdims=True), grid.shape).copy(), False, None, False),
            "first_only": (grid, True, None, True),
        }
        choices = {}
        for offset, (name, (variant_grid, history, mask, first_only)) in enumerate(variants.items()):
            progress(f"fit_{name}")
            factory = make(variant_grid, first_only=first_only)
            context = baseline(raw_env, variant_grid, replay, settings | {"first_only": first_only},
                               "queue_aware", objective, search_cfg["anchor_wait_weight"], name != "no_congestion")
            ml = ml_actor(context)
            fitted = fit_search(factory, context, ml, training, validation, settings, search_cfg, objective,
                                np.random.SeedSequence([seed, offset, 71]), history, mask)
            pick = fitted["selected"]
            actor = ml if pick is None else fitted["actors"][pick]
            scored = detailed_metrics(factory, testing, {name: actor, f"{name}_ML": ml}, n)
            rows.update(scored)
            choices[name] = {"selected": pick, "forced": fitted["forced"], "fallback": pick is None}
        for arm in variants:
            if arm != "full":
                contrasts[f"full_minus_{arm}"] = {"full": 1.0, arm: -1.0}
        contrasts["full_minus_supervised_ML"] = {"full": 1.0, "full_ML": -1.0}
        design["choices"] = choices

    elif domain == "bpisim":
        choices = {}
        for ready in (False, True):
            label = "ready" if ready else "marker"
            progress(f"fit_{label}")
            factory = make(grid, ready=ready)
            context = baseline(raw_env, grid, replay, settings, "queue_aware", "elapsed",
                               search_cfg["anchor_wait_weight"])
            ml = ml_actor(context)
            fitted = fit_search(factory, context, ml, training, validation, settings, search_cfg, "elapsed",
                                np.random.SeedSequence([seed, int(ready), 81]))
            selected = ml if fitted["selected"] is None else fitted["actors"][fitted["selected"]]
            forced = fitted["actors"][fitted["forced"]]
            static_mask = [1] + [0] * (len(search_cfg["initial_spread"]) - 1)
            static = fit_search(factory, context, ml, training, validation, settings, search_cfg, "elapsed",
                                np.random.SeedSequence([seed, int(ready), 82]), mask=static_mask)
            static_actor = ml if static["selected"] is None else static["actors"][static["selected"]]
            rows.update(detailed_metrics(factory, testing, {f"{label}_ML": ml, f"{label}_selected": selected,
                                                             f"{label}_forced": forced,
                                                             f"{label}_equal_static": static_actor}, n))
            choices[label] = {"selected": fitted["selected"], "forced": fitted["forced"],
                              "fallback": fitted["selected"] is None}
        contrasts = {
            "marker_selected_minus_ML": {"marker_selected": 1.0, "marker_ML": -1.0},
            "marker_forced_minus_ML": {"marker_forced": 1.0, "marker_ML": -1.0},
            "marker_selected_minus_equal_static": {"marker_selected": 1.0, "marker_equal_static": -1.0},
            "ready_selected_minus_ML": {"ready_selected": 1.0, "ready_ML": -1.0},
            "timing_interaction": {"ready_selected": 1.0, "ready_ML": -1.0,
                                   "marker_selected": -1.0, "marker_ML": 1.0},
        }
        design["choices"] = choices

    else:
        progress("fit_tail_controller")
        factory = make(grid)
        context = baseline(raw_env, grid, replay, settings, "queue_aware", "tail_queue",
                           search_cfg["anchor_wait_weight"])
        ml = ml_actor(context)
        fitted = fit_search(factory, context, ml, training, validation, settings, search_cfg, "tail_queue",
                            np.random.SeedSequence([seed, 91]))
        selections = {
            "mean_blend": choose_blend(fitted, settings, search_cfg, .75, False),
            "safe_050": choose_blend(fitted, settings, search_cfg, .50, True),
            "safe_075": choose_blend(fitted, settings, search_cfg, .75, True),
            "safe_100": choose_blend(fitted, settings, search_cfg, 1.00, True),
        }
        proposed = selections["safe_075"]
        if proposed is None:
            proposed = (fitted["forced"], 0.0, fitted["reference"])
        actors = {"ML": ml, "raw": fitted["actors"][proposed[0]]}
        component = detailed_metrics(factory, testing, actors, n)
        rows.update(component)
        for name, picked in selections.items():
            rows[name] = component["ML"] if picked is None else weighted_rows(
                component["ML"], detailed_metrics(factory, testing, {"candidate": fitted["actors"][picked[0]]}, n)["candidate"], picked[1])
        no_wait_context = baseline(raw_env, grid, replay, settings, "queue_aware", "tail_queue",
                                   search_cfg["anchor_wait_weight"], tail_state=False)
        mask = np.ones(len(search_cfg["initial_spread"])); mask[-1] = 0.0
        no_wait = fit_search(factory, no_wait_context, ml, training, validation, settings, search_cfg, "tail_queue",
                             np.random.SeedSequence([seed, 92]), mask=mask)
        no_wait_pick = choose_blend(no_wait, settings, search_cfg, .75, True)
        rows["no_accrued_wait"] = component["ML"] if no_wait_pick is None else weighted_rows(
            component["ML"], detailed_metrics(factory, testing, {"candidate": no_wait["actors"][no_wait_pick[0]]}, n)["candidate"], no_wait_pick[1])
        queue_context = baseline(raw_env, grid, replay, settings, "queue_aware", "queue",
                                 search_cfg["anchor_wait_weight"])
        queue_fit = fit_search(factory, queue_context, ml, training, validation, settings, search_cfg, "queue",
                               np.random.SeedSequence([seed, 93]))
        queue_pick = queue_fit["selected"] or queue_fit["forced"]
        rows["mean_queue_objective"] = detailed_metrics(factory, testing,
            {"candidate": queue_fit["actors"][queue_pick]}, n)["candidate"]
        burst_factory = make(grid, arrivals=lambda values: cluster_arrivals(values,
                             config["burst_size"], config["burst_gap_factor"]))
        burst = detailed_metrics(burst_factory, testing, actors, n)
        rows["burst_ML"], rows["burst_raw"] = burst["ML"], burst["raw"]
        rows["burst_safe_075"] = weighted_rows(burst["ML"], burst["raw"], proposed[1])
        contrasts = {
            "safe_075_minus_ML": {"safe_075": 1.0, "ML": -1.0},
            "safe_075_minus_raw": {"safe_075": 1.0, "raw": -1.0},
            "safe_075_minus_mean_blend": {"safe_075": 1.0, "mean_blend": -1.0},
            "safe_075_minus_no_accrued_wait": {"safe_075": 1.0, "no_accrued_wait": -1.0},
            "safe_075_minus_mean_queue_objective": {"safe_075": 1.0, "mean_queue_objective": -1.0},
            "burst_safe_075_minus_ML": {"burst_safe_075": 1.0, "burst_ML": -1.0},
        }
        design["selections"] = {name: None if value is None else {"round": value[0], "weight": value[1]}
                                for name, value in selections.items()}

    design.update({"fields": FIELDS, "contrasts": contrasts, "seed": seed, "source": str(source),
                   "source_sha256": {str(path): sha256(path.read_bytes()) for path in
                                     (source / "freeze.json", source / "predictors.joblib")},
                   "split": {"fit_cases": len(fit), "heldout_cases": len(historical),
                             "boundary": boundary, "excluded": excluded},
                   "provenance": dict(provenance)})
    save(folder / "design.json", design)
    np.savez_compressed(folder / "paired.npz", **rows)
