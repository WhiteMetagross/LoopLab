"""Paired ablations of routing timing, policy inputs, and specialist effort."""

from pathlib import Path

import numpy as np

from scripts import simfactory_tradeoff_replay as blends
from scripts import specialist_slot_search as search
from scripts.audit_routing_rewards import ml_plan
from scripts.specialist_slots_pilot import SlotEpisode, TAIL_FIELDS, baseline, finish, ml_actor
from src.domain_loop_data import acquire, chronological_split, load_cases, sha256
from src.domain_loop_simulations import save
from src.domain_loop_simulator import ACTIONS, OUTCOMES, LoopEnvironment, calibrate
from src.domain_loop_training import read_replay, scenarios


FIELDS = (*TAIL_FIELDS, "pre_service", "residual")


def assess(factory, workloads, controls, settings, include_repeat_slots=False):
    result = {}
    for name, actor in controls.items():
        rows = []
        for workload in workloads:
            slots = 0 if name == "FIFO" else settings["slots"]
            episode = SlotEpisode(factory(), workload, slots, settings.get("tail_fraction", .1), settings.get("first_only", False))
            values = finish(episode, actor)
            before = sum(row.service_ready - row.ready for row in episode.env.records)
            residual = sum(row.end - row.service_end for row in episode.env.records)
            np.testing.assert_allclose(values[1], before + values[2] + values[5] + residual, atol=1e-6)
            if values[6] != slots:
                raise ValueError("Fixed specialist quota violated")
            totals = np.r_[values, before, residual]
            if include_repeat_slots:
                repeat_slots = sum(row.attempt > 1 and row.action == "specialist" for row in episode.env.records)
                if settings.get("first_only") and repeat_slots:
                    raise ValueError("First request policy used a later specialist slot")
                totals = np.r_[totals, repeat_slots]
            rows.append(totals / episode.size)
        result[name] = np.asarray(rows)
    return result


def mechanism_grid(env, bins):
    rows = np.empty((2, bins, 2, 4))
    for stage in (1, 2):
        timing = [np.mean([(t.service, t.service + t.lag + t.before_service)
                          for t in env.law.timing_pool(stage, outcome)], axis=0) for outcome in OUTCOMES]
        for index in range(bins):
            shift = env.difficulty_span * ((index + .5) / bins - .5)
            p = env.law.probability(stage, env.scenario_repeat, shift)
            work, elapsed = p @ np.asarray(timing)
            added = (env.multiplier - 1) * work
            rows[stage - 1, index] = [[p[0], elapsed, work, sum(p[2:])],
                                      [p[0] * (1 - env.effect), elapsed + added, work + added, sum(p[2:])]]
    return rows


def bpi(source, make, workloads, settings):
    frozen = blends.read(source / "freeze.json")
    grid = np.load(source / "forecast_grid.npy")
    replay = read_replay(source / "ml_fit" / "transitions.jsonl")
    name = frozen["best_ml"]
    env = make()
    ml = ml_actor(baseline(env, grid, replay, settings, "queue_aware", "elapsed", float(name.split("_")[-1])))
    lookahead = ml_plan(env, mechanism_grid(env, settings["difficulty_bins"]),
                        (1., 1. / env.law.time_scale, 0., .5 / env.law.mean_service, 2.), True)
    controls = {"FIFO": lambda _: 0, "ML": ml, "Lookahead": lambda ep: ACTIONS.index(lookahead(ep.state))}
    rows = {}
    for timing in ("marker", "ready"):
        scores = assess(lambda: make(ready=timing == "ready"), workloads, controls, settings)
        rows.update({f"{name}_{timing}": values for name, values in scores.items()})
    for field in ("extra_attempts", "elapsed", "work", "unresolved", "pre_service", "residual"):
        np.testing.assert_allclose(rows["FIFO_marker"][:, FIELDS.index(field)], rows["FIFO_ready"][:, FIELDS.index(field)], atol=1e-7)
    return rows, {"ml": name}, {key: controls[key] for key in ("ML", "Lookahead")}


def factory(source, make, workloads, settings):
    frozen = blends.read(source / "freeze.json")
    choice = blends.read(source / "blend_freeze.json")
    controls = blends.make_controls(frozen, np.load(source / "forecast_grid.npy"),
                                    read_replay(source / "ml_fit" / "transitions.jsonl"), make)
    validation, reference = frozen["validation"], choice["reference"]
    caps = blends.validation_caps(validation, reference, settings["limits"])
    feasible = [name for name, row in validation.items() if name != "FIFO"
                and all(row[field] <= cap + 1e-9 for field, cap in caps.items())]
    picks = {"control_blend": choice["ml_mix"], "hybrid_blend": choice["hybrid_mix"],
             "original_reference": {reference: 1.}}
    for label, hybrid in (("control_single", False), ("hybrid_single", True)):
        best = min((name for name in feasible if name.startswith("RL_") == hybrid),
                   key=lambda name: validation[name]["tail_queue"], default=None)
        picks[label] = {best: 1.} if best else None
    used = {name for weights in picks.values() if weights for name in weights}
    components = assess(make, workloads, {name: controls[name] for name in sorted(used | {"FIFO"})}, settings)
    rows = {label: blends.weighted_rows(weights, components) for label, weights in picks.items() if weights}
    rows["FIFO"] = components["FIFO"]
    return rows, picks, controls, components


def claims(folder, index, namespace, arms, settings, make, workloads):
    full = folder / "full"
    choices, controls = {}, {}
    for arm in arms:
        config = folder / f"{arm}.json"
        save(config, settings | {"ablation": arm})
        path = folder / arm
        search.run(path, config, index, namespace, "insurance", prepare_only=True,
                   fit_source=None if arm == "full" else full)
        if not (path / "freeze.json").is_file():
            raise ValueError(f"SimClaims arm {arm} has no feasible supervised comparator")
        frozen = blends.read(path / "freeze.json")
        choices[arm] = frozen["selected_rl"]
        if arm == "full":
            replay = read_replay(full / "ml_fit" / "transitions.jsonl")
            grid = np.load(full / "forecast_grid.npy")
            best = frozen["best_ml"]
            controls["ML"] = ml_actor(baseline(make(), grid, replay, settings, "queue_aware", "repeat", float(best.split("_")[-1])))
            choices["ML"] = best
        if choices[arm] is None:
            controls[arm] = controls["ML"]
            continue
        anchor = baseline(make(), np.load(path / "forecast_grid.npy"), replay, settings, "queue_aware",
                          "repeat", settings.get("anchor_wait_weight", .04), arm != "no_congestion")
        controls[arm] = search.residual_actor(anchor, frozen["weights"][choices[arm]], settings["slots"], arm != "no_loop_history")
    save(folder / "choice.json", choices)
    return assess(make, workloads, controls | {"FIFO": lambda _: 0}, settings), controls


def sensitivity(make, workloads, settings, multipliers, controls, picks):
    rows = {}
    used = {name for weights in picks.values() for name in weights}
    for multiplier in multipliers:
        values = assess(lambda: make(multiplier=multiplier), workloads,
                        {name: controls[name] for name in sorted(used)}, settings)
        rows.update({f"{label}_{multiplier:g}": blends.weighted_rows(weights, values) for label, weights in picks.items()})
    return rows


def run(folder, index, seed, domain, config, profile, progress):
    sim = blends.read(Path(config["domain_config"]))
    spec = sim["domains"][domain]
    raw = acquire(spec, sim["maximum_download_bytes"])
    cases, _ = load_cases(raw, domain, spec)
    fit, historical, boundary, excluded = chronological_split(cases, sim["fit_fraction"])
    law = calibrate(fit, spec, sim)
    paper = Path(config["paper_run"])
    leaf = f"seed_{index}"
    source = {"bpisim": paper / "bpisim_turnaround" / leaf,
              "manufacturing": Path(config["factory_run"]) / leaf}.get(domain)
    if source is None:
        settings = blends.read(Path("configs/specialist_slot_search.json")) | profile["claims"]
        artifacts = ()
    else:
        frozen = blends.read(source / "freeze.json")
        if frozen["seed"] != seed:
            raise ValueError("Frozen routing seed does not match")
        settings = frozen["settings"]
        artifacts = ("freeze.json", "forecast_grid.npy", "ml_fit/transitions.jsonl")
        if domain == "manufacturing":
            artifacts = (*artifacts, "blend_freeze.json")

    def make(ready=False, multiplier=sim["specialist_service_multiplier"]):
        return LoopEnvironment(law, spec, sim | {"specialist_service_multiplier": multiplier}, settings["effect"],
                               scenario=True, route_when_ready=ready)

    seed_stream = np.random.SeedSequence([seed, profile["namespace"], 0])
    workloads = scenarios(seed_stream, law, spec, settings, profile["workloads"])
    if domain == "bpisim":
        progress("decision_timing")
        rows, choice, controls = bpi(source, make, workloads, settings)
        picks = {name: {name: 1.} for name in controls}
        contrasts = {f"{policy}_ready_minus_marker": {f"{policy}_ready": 1, f"{policy}_marker": -1}
                     for policy in ("ML", "Lookahead")}
        contrasts |= {f"lookahead_minus_ml_{timing}": {f"Lookahead_{timing}": 1, f"ML_{timing}": -1}
                      for timing in ("marker", "ready")}
    elif domain == "manufacturing":
        progress("single_and_blended")
        rows, choice, controls, components = factory(source, make, workloads, settings)
        np.savez_compressed(folder / "factory_components.npz", **components)
        picks = {name: choice[name] for name in ("control_blend", "hybrid_blend")}
        contrasts = {f"{a}_minus_{b}": {a: 1, b: -1} for a, b in (
            ("hybrid_single", "control_single"), ("hybrid_blend", "hybrid_single"),
            ("control_blend", "control_single"), ("hybrid_blend", "control_blend"),
            ("hybrid_blend", "original_reference"))}
    else:
        progress("claims_policy_search")
        rows, controls = claims(folder, index, profile["namespace"], config["claims_arms"], settings, make, workloads)
        choice = blends.read(folder / "choice.json")
        picks = {name: {name: 1.} for name in ("ML", "full")}
        contrasts = {f"full_minus_{arm}": {"full": 1, arm: -1} for arm in (*config["claims_arms"][1:], "ML")}
    progress("specialist_duration_sweep")
    multipliers = config["specialist_multipliers"]
    rows.update(sensitivity(make, workloads, settings, multipliers, controls, picks))
    center = sim["specialist_service_multiplier"]
    for label in picks:
        for multiplier in multipliers:
            if multiplier != center:
                arm, ref = f"{label}_{multiplier:g}", f"{label}_{center:g}"
                contrasts[f"{arm}_minus_{ref}"] = {arm: 1, ref: -1}
    np.savez_compressed(folder / "workloads.npz", arrivals=np.asarray([row[0] for row in workloads]),
                        noise=np.asarray([row[1] for row in workloads], dtype=np.int64))
    save(folder / "design.json", {"fields": FIELDS, "contrasts": contrasts, "choice": choice, "settings": settings,
                                  "source": str(source) if source else None, "data": spec,
                                  "source_sha256": {name: sha256((source / name).read_bytes()) for name in artifacts},
                                  "split": {"fit_cases": len(fit), "later_cases": len(historical),
                                            "boundary": boundary, "excluded_overlap": excluded}})
    np.savez_compressed(folder / "paired.npz", **rows)
