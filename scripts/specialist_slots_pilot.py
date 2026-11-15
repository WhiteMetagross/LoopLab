"""Compare supervised slot allocation with paired rollout policy improvement."""

import argparse
from collections import Counter
from copy import copy
from importlib.metadata import version
import json
from math import ceil
from pathlib import Path
import platform
import signal
from time import perf_counter

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from scripts.audit_routing_rewards import feasible, fit_grids, gaps
from src.benchmark_models import estimator, predict, seeds
from src.domain_loop_data import acquire, chronological_split, load_cases, selected_domains, sha256
from src.domain_loop_simulations import generate, save
from src.domain_loop_simulator import ACTIONS, LoopEnvironment, calibrate
from src.domain_loop_training import difficulty_bin, read_replay, scenarios


FIELDS = ("extra_attempts", "elapsed", "work", "specialist_work", "unresolved", "queue", "slots")
TAIL_FIELDS = (*FIELDS, "tail_queue")
OBJECTIVES = ("repeat", "queue")


def upper_tail_mean(values, fraction):
    values = np.asarray(values, dtype=float)
    start = len(values) - ceil(len(values) * fraction)
    return float(np.partition(values, start)[start:].mean())


class SlotEpisode:
    def __init__(self, env, workload, slots, tail_fraction=None, first_only=False):
        self.env, self.slots, self.seen, self.closed = env, slots, 0, 0
        self.size = len(workload[0])
        self.tail_fraction = tail_fraction
        self.first_only = first_only
        self.accrued_wait = np.zeros(self.size) if tail_fraction is not None else None
        self.env.reset(*workload)
        self._observe()

    @property
    def state(self):
        return self.env.observation

    def _observe(self):
        if self.state is not None and self.state.attempt == 1:
            self.seen += 1

    def forced(self):
        if self.slots == 0 or self.state.attempt == self.env.limit:
            return 0
        if self.first_only and self.state.attempt > 1:
            return 0
        if self.state.attempt == 1 and self.slots >= self.size - self.seen + 1:
            return 1
        return None

    def step(self, action):
        self.slots -= action
        assert self.slots >= 0
        _, _, _, completed = self.env.step(ACTIONS[action])
        self.closed += sum(item.outcome != "repeat" for item in completed)
        if self.accrued_wait is not None:
            for item in completed:
                self.accrued_wait[item.case] += item.service_start - item.service_ready
        self._observe()

    def fork(self):
        branch = copy(self)
        branch.env = copy(self.env)
        for name in ("calendar", "records", "sequence"):
            setattr(branch.env, name, copy(getattr(self.env, name)))
        for name in ("free", "active"):
            setattr(branch.env, name, {key: value.copy() for key, value in getattr(self.env, name).items()})
        if self.accrued_wait is not None:
            branch.accrued_wait = self.accrued_wait.copy()
        return branch

    def result(self):
        rows = self.env.case_rows()
        specialist = [row for row in self.env.records if row.action == "specialist"]
        values = [sum(row["attempts"] - 1 for row in rows), sum(row["elapsed"] for row in rows),
                  sum(row["work"] for row in rows), sum(row.service_end - row.service_start for row in specialist),
                  sum(row["outcome"] != "resolved" for row in rows), sum(row["queue_wait"] for row in rows), len(specialist)]
        if self.tail_fraction is not None:
            values.append(self.size * upper_tail_mean([row["queue_wait"] for row in rows], self.tail_fraction))
        return np.asarray(values, float)


def finish(ep, policy):
    while ep.state is not None:
        action = ep.forced()
        ep.step(policy(ep) if action is None else action)
    assert ep.slots == 0 and ep.closed == ep.seen == ep.size
    return ep.result()


def opportunity_values(gains, horizon, slots):
    values = np.zeros((horizon + 1, slots + 1))
    for t in range(1, horizon + 1):
        values[t, 1:] = np.maximum(values[t - 1, 1:], values[t - 1, :-1] + gains[:, None]).mean(axis=0)
    return values


def baseline(env, grid, replay, cfg, mode, objective, wait_weight_override=None, congestion=True, tail_state=True):
    risk, elapsed, work, unresolved = np.moveaxis(grid, -1, 0)
    expected_loops = sum(risk[1, :, 0] ** j for j in range(env.limit - 1))
    repeat_gain = (risk[..., 0] - risk[..., 1]) * expected_loops[None, :]
    effort_gain = (work[..., 0] - work[..., 1]) / env.law.mean_service
    age_gain = (elapsed[..., 0] - elapsed[..., 1]) / env.law.time_scale
    resolved_gain = unresolved[..., 0] - unresolved[..., 1]
    tail_aware = objective == "tail_queue" and tail_state
    if objective in ("repeat", "tail_queue"):
        gain = repeat_gain + .03 * effort_gain + .05 * age_gain + .15 * resolved_gain
        wait_weight = .04 if mode == "queue_aware" else 0.
    elif objective == "elapsed":
        continuation = elapsed[1, :, 0] / env.law.time_scale
        gain = age_gain + repeat_gain * continuation[None, :]
        wait_weight = env.law.mean_service / env.law.time_scale
    else:
        gain = .7 * repeat_gain + .05 * effort_gain + .02 * age_gain + .15 * resolved_gain
        wait_weight = 1. if mode == "queue_aware" else .5
    if wait_weight_override is not None:
        wait_weight = wait_weight_override
    bins = grid.shape[1]
    observed = []
    past_wait = {}
    for row, result in replay:
        s = row["state"]
        if cfg.get("first_only") and s["attempt"] > 1:
            continue
        wait = (s["standard_wait_estimate"] - s["specialist_wait_estimate"]) / env.law.mean_service if congestion else 0.
        if tail_aware:
            key = (row["workload"], s["case"])
            prior = past_wait.get(key, 0.)
            burden = (prior + max(s["standard_wait_estimate"], s["specialist_wait_estimate"])) / env.law.mean_service
            wait *= 1 + min(2., burden / 4.)
            past_wait[key] = prior + result["service_start"] - result["service_ready"]
        observed.append(gain[min(s["attempt"], 2) - 1, difficulty_bin(s["difficulty_signal"], bins)] + wait_weight * wait)
    opportunities_per_case = len(replay) / (cfg["ml_fit_workloads"] * cfg["cases_per_workload"])
    price = opportunity_values(np.asarray(observed), 2 * cfg["cases_per_workload"], cfg["slots"])
    static_quantile = 0.0 if opportunities_per_case <= 0 else np.clip(
        1 - cfg["slots"] / cfg["cases_per_workload"] / opportunities_per_case, 0.0, 1.0
    )
    static_price = float(np.quantile(observed, static_quantile))

    def context(ep):
        s = ep.state
        stage, d = min(s.attempt, 2) - 1, difficulty_bin(s.difficulty_signal, bins)
        a, b = (s.standard_wait_estimate / env.law.mean_service, s.specialist_wait_estimate / env.law.mean_service) if congestion else (0., 0.)
        wait = a - b
        if tail_aware:
            burden = ep.accrued_wait[s.case] / env.law.mean_service + max(a, b)
            wait *= 1 + min(2., burden / 4.)
        score = gain[stage, d] + wait_weight * wait
        future = ep.size - ep.seen if ep.first_only else round((ep.size - ep.seen) * opportunities_per_case + (ep.seen - ep.closed) * .35)
        future = min(future, len(price) - 1)
        shadow = static_price if mode == "fixed_threshold" else price[future, ep.slots] - price[future, ep.slots - 1]
        advantage = score - shadow
        return (advantage, score, shadow, risk[stage, d, 0], risk[stage, d, 1], work[stage, d, 0] / env.law.mean_service,
                work[stage, d, 1] / env.law.mean_service, a, b, s.attempt, s.difficulty_signal,
                ep.slots, ep.size - ep.seen, ep.seen - ep.closed, s.case_age / env.law.time_scale)

    return context


def assess(factory, workloads, policies, cfg):
    return {name: np.array([finish(SlotEpisode(factory(), workload, 0 if name == "FIFO" else cfg["slots"],
                                               cfg.get("tail_fraction"), cfg.get("first_only", False)), policy)
                            / cfg["cases_per_workload"] for workload in workloads]) for name, policy in policies.items()}


def ml_actor(context):
    return lambda ep: int(context(ep)[0] > 0.)


def targets(values, env):
    repeat, elapsed, work, _, unresolved, queue, _ = values
    return np.array([repeat + .05 * elapsed / env.law.time_scale + .03 * work / env.law.mean_service + .15 * unresolved,
                     queue / env.law.mean_service + .7 * repeat + .02 * elapsed / env.law.time_scale + .05 * work / env.law.mean_service + .15 * unresolved])


def fit_residual(factory, workloads, context, objective, cfg, rng, seed):
    x, y = [], []
    objective_index = OBJECTIVES.index(objective)
    policy = ml_actor(context)
    for workload in workloads:
        ep = SlotEpisode(factory(), workload, cfg["slots"])
        candidates = []
        while ep.state is not None:
            action = ep.forced()
            if action is None:
                features = context(ep)
                candidates.append((ep.fork(), features))
                action = int(features[0] > 0.)
            ep.step(action)
        baseline_total = ep.result()
        chosen = rng.choice(len(candidates), min(cfg["rollout_decisions_per_workload"], len(candidates)), replace=False)
        for idx in chosen:
            branch, features = candidates[idx]
            chosen_action = int(features[0] > 0.)
            branch.step(1 - chosen_action)
            alternative = finish(branch, policy)
            advantage = (targets(baseline_total - alternative, ep.env) * (1 - 2 * chosen_action))[objective_index]
            x.append(features)
            y.append(advantage - features[0])
    model = estimator("CatBoost", "effort", seed, cfg["trees"]).fit(np.asarray(x), np.asarray(y))
    return model, len(x)


def compare(rows, baseline_rows, fifo, cfg, draws, objective):
    delta = rows - baseline_rows
    limit = cfg["limits"]
    margin = np.column_stack((gaps(rows, fifo, limit), gaps(rows, baseline_rows, limit),
                              rows[:, 3] - limit["specialist_share"] * fifo[:, 2]))
    diff_ci = np.quantile(delta[draws].mean(axis=1), [.025, .975], axis=0)
    constraints = (np.quantile(margin[draws].mean(axis=1), .95, axis=0) <= 0).all()
    if objective == "repeat":
        improvement = delta[:, 0].mean() <= -cfg["minimum_repeat_reduction"] and diff_ci[1, 0] < 0
    elif objective == "elapsed":
        improvement = (delta[:, 1].mean() <= -cfg["minimum_elapsed_reduction_minutes"] and diff_ci[1, 1] < 0
                       and diff_ci[1, 0] <= cfg["repeat_noninferiority"])
    elif objective == "tail_queue":
        improvement = (delta[:, 7].mean() <= -cfg["minimum_tail_queue_reduction_fraction"] * baseline_rows[:, 7].mean()
                       and diff_ci[1, 7] < 0 and diff_ci[1, 0] <= cfg["repeat_noninferiority"])
    else:
        improvement = (delta[:, 5].mean() <= -cfg["minimum_queue_reduction_fraction"] * baseline_rows[:, 5].mean()
                       and diff_ci[1, 5] < 0 and diff_ci[1, 0] <= cfg["repeat_noninferiority"])
    fields = TAIL_FIELDS if objective == "tail_queue" else FIELDS
    return {"delta": dict(zip(fields, delta.mean(axis=0), strict=True)),
            "paired_ci95": dict(zip(fields, diff_ci.T.tolist(), strict=True)),
            "operating_bounds_pass": bool(constraints), "objective_pass": bool(improvement), "passed": bool(constraints and improvement)}


def run_domain(path, law, spec, sim, cfg, seed):
    fit, rollout, validation, evaluation, sampling, bootstrap = np.random.SeedSequence(seed).spawn(6)
    path.mkdir()
    logs = path / "ml_fit"
    logs.mkdir()
    generate(law, spec, sim, fit, logs, cfg["load"], cfg["effect"], cfg["ml_fit_workloads"], cfg["cases_per_workload"], .5)
    replay = read_replay(logs / "transitions.jsonl")
    grid = fit_grids(replay, cfg, seed)
    factory = lambda: LoopEnvironment(law, spec, sim, cfg["effect"], scenario=True)
    batches = {key: scenarios(child, law, spec, cfg, cfg[f"{key}_workloads"])
               for key, child in (("rollout", rollout), ("validation", validation), ("evaluation", evaluation))}
    contexts = {f"ML_{objective}_{mode}": baseline(factory(), grid, replay, cfg, mode, objective)
                for objective in OBJECTIVES for mode in ("fixed_threshold", "adaptive", "queue_aware")}
    policies = {"FIFO": lambda _: 0, "Spend_first": lambda _: 1,
                "Shorter_queue": lambda ep: int(ep.state.specialist_wait_estimate < ep.state.standard_wait_estimate),
                **{name: ml_actor(ctx) for name, ctx in contexts.items()}}
    val = assess(factory, batches["validation"], policies, cfg)
    admissible = [name for name in contexts if feasible(val[name], val["FIFO"], cfg["limits"])]
    if not admissible:
        save(path / "results.json", {"seed": seed, "validation": summarize(val), "selected": {}, "status": "no_feasible_ml"})
        print(f"{spec['name']}: no feasible ML policy on validation", flush=True)
        return
    repeat_ml = min(admissible, key=lambda name: val[name][:, 0].mean())
    queue_ml = min((name for name in admissible if val[name][:, 0].mean() <= val[repeat_ml][:, 0].mean() + cfg["repeat_noninferiority"]),
                   key=lambda name: val[name][:, 5].mean())
    selected_ml = {"repeat": repeat_ml, "queue": queue_ml}
    rng = np.random.default_rng(sampling)
    actors, fitted = {}, {}
    for objective, matched in selected_ml.items():
        context = contexts[matched]
        critic, n = fit_residual(factory, batches["rollout"], context, objective, cfg, rng, seed)
        fitted[objective] = (critic, n)
        for weight in cfg["residual_weights"]:
            def actor(ep, ctx=context, model=critic, w=weight):
                features = ctx(ep)
                return int(features[0] + w * predict(model, np.asarray([features]))[0] > 0.)

            actors[f"RL_{objective}_{weight:g}"] = actor
    joblib.dump((grid, fitted), path / "models.joblib")
    val.update(assess(factory, batches["validation"], actors, cfg))
    selected = {}
    for objective, matched in selected_ml.items():
        index = FIELDS.index("extra_attempts" if objective == "repeat" else "queue")
        candidates = [name for name in actors if name.startswith(f"RL_{objective}_")
                      and feasible(val[name], val["FIFO"], cfg["limits"], val[matched])
                      and (objective != "queue" or val[name][:, 0].mean() <= val[matched][:, 0].mean() + cfg["repeat_noninferiority"])]
        best = min(candidates, default=None, key=lambda name: val[name][:, index].mean())
        selected[objective] = best if best and val[best][:, index].mean() < val[matched][:, index].mean() else None
    freeze = {"seed": seed, "selected_ml": selected_ml, "selected": selected, "validation": summarize(val),
              "pairs_fit": {name: entry[1] for name, entry in fitted.items()}}
    save(path / "freeze.json", freeze)
    print(f"{spec['name']}: frozen ML {selected_ml}, RL {selected}", flush=True)
    final = assess(factory, batches["evaluation"], policies | actors, cfg)
    assert all(np.allclose(rows[:, -1] * cfg["cases_per_workload"], 0 if name == "FIFO" else cfg["slots"])
               for name, rows in final.items())
    count = cfg["evaluation_workloads"]
    draws = np.random.default_rng(bootstrap).integers(count, size=(cfg["bootstrap_draws"], count))
    comparisons = {name: compare(final[name], final[selected_ml[name.split('_')[1]]], final["FIFO"], cfg, draws, name.split('_')[1])
                   for name in actors}
    np.savez_compressed(path / "paired_workloads.npz", **final)
    save(path / "results.json", freeze | {"evaluation": summarize(final), "comparisons": comparisons,
                                          "passed": {objective: bool(name and comparisons[name]["passed"]) for objective, name in selected.items()}})


def summarize(rows):
    return {key: dict(zip(TAIL_FIELDS if values.shape[1] == len(TAIL_FIELDS) else FIELDS,
                          values.mean(axis=0), strict=True)) for key, values in rows.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--config", default="configs/specialist_slots_pilot.json", type=Path)
    args = parser.parse_args()
    start = perf_counter()
    cfg = json.loads(args.config.read_text())
    config_path = Path("configs/domain_loop_simulations.json")
    sim = json.loads(config_path.read_text())
    assert 0 < cfg["slots"] <= cfg["cases_per_workload"]
    signal.alarm(cfg["stage_cap_seconds"])
    seed = seeds(sim["master_seed"], 1)[0]
    args.out.mkdir(parents=True)
    files = (args.config, config_path, Path(__file__), Path("src/domain_loop_simulator.py"),
             Path("src/domain_loop_data.py"), Path("src/bpisim_data.py"), Path("src/domain_loop_simulations.py"),
             Path("src/domain_loop_training.py"), Path("scripts/audit_routing_rewards.py"), Path("src/benchmark_models.py"))
    manifest = {"seed": seed, "settings": cfg, "python": platform.python_version(), "device": "CPU",
                "versions": {name: version(name) for name in ("numpy", "catboost", "scikit-learn")},
                "code_and_config_sha256": {str(p): sha256(p.read_bytes()) for p in files}}
    save(args.out / "manifest.json", manifest)
    with threadpool_limits(limits=1):
        for index, domain, spec in selected_domains(sim, "all"):
            if domain not in cfg["domains"]:
                continue
            raw = acquire(spec, sim["maximum_download_bytes"])
            cases, _ = load_cases(raw, domain, spec)
            fit, _, _, _ = chronological_split(cases, sim["fit_fraction"])
            run_domain(args.out / domain, calibrate(fit, spec, sim), spec, sim, cfg, seeds([seed, index, 297], 1)[0])
    save(args.out / "manifest.json", manifest | {"status": "complete", "seconds": perf_counter() - start})
    signal.alarm(0)


if __name__ == "__main__":
    main()
