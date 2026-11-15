"""One-seed reward comparison with shared replay and paired simulator workloads."""

import argparse
import json
from pathlib import Path
import platform
from time import perf_counter

import numpy as np
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from src import paper_rl
from src.benchmark_models import estimator, predict, seeds
from src.domain_loop_data import acquire, chronological_split, load_cases, selected_domains, sha256
from src.domain_loop_simulations import generate, save
from src.domain_loop_simulator import ACTIONS, LoopEnvironment, Observation, calibrate, run_policy
from src.domain_loop_training import MASK, attach_returns, difficulty_bin, ml_features, ml_rule, read_replay, representation, scenarios
from src.runtime import enable_shared_runtime


FIELDS = ("extra_attempts", "elapsed", "work", "specialist_work", "unresolved")


def components(event):
    work = event["service_end"] - event["service_start"]
    return np.array([event["outcome"] == "repeat", event["end"] - event["ready"], work,
                     work * (event["action"] == "specialist"),
                     event["outcome"] in {"negative", "unobserved", "cap"}])


def measure(env, workloads, policy):
    rows = []
    for arrivals, noise in workloads:
        cases = run_policy(env, arrivals, noise, policy)
        values = np.array([sum(row["attempts"] - 1 for row in cases), sum(row["elapsed"] for row in cases),
                           sum(row["work"] for row in cases),
                           sum(row.service_end - row.service_start for row in env.records if row.action == "specialist"),
                           sum(row["outcome"] != "resolved" for row in cases)])
        np.testing.assert_allclose(sum(components(vars(row)) for row in env.records), values, atol=1e-7)
        rows.append(values / len(cases))
    return np.asarray(rows)


def gaps(candidate, reference, limits):
    return np.column_stack((candidate[:, 1] - limits["elapsed_ratio"] * reference[:, 1],
                            candidate[:, 2] - limits["work_ratio"] * reference[:, 2],
                            candidate[:, 4] - reference[:, 4] - limits["resolution_loss"]))


def feasible(candidate, fifo, limits, ml=None):
    reference_ok = (gaps(candidate, fifo, limits).mean(axis=0) <= 0).all()
    capacity_ok = candidate[:, 3].mean() <= limits["specialist_share"] * fifo[:, 2].mean()
    return bool(reference_ok and capacity_ok and (ml is None or (gaps(candidate, ml, limits).mean(axis=0) <= 0).all()))


def fit_grids(replay, cfg, seed, history=True):
    bins = cfg["difficulty_bins"]
    x = np.array([ml_features(Observation(**row["state"]), row["action"], bins) for row, _ in replay])
    z = np.array([(s, b, a) for s in (1, 2) for b in range(bins) for a in range(len(ACTIONS))])
    if not history:
        x[:, 0] = z[:, 0] = 1
    y = np.array([components(event)[[0, 1, 2, 4]] - [0, event["service_start"] - event["service_ready"], 0, 0]
                  for _, event in replay])
    estimates = []
    for i, target in enumerate(y.T):
        binary = i in (0, 3)
        if np.ptp(target) == 0:
            estimates.append(np.full(len(z), target[0]))
            continue
        model = estimator("CatBoost", "risk" if binary else "effort", seed, cfg["trees"]).fit(x, target)
        prediction = predict(model, z, probability=binary)
        estimates.append(prediction[:, 1] if binary else np.maximum(0., prediction))
    return np.column_stack(estimates).reshape(2, bins, len(ACTIONS), 4)


def ml_plan(env, grid, weights, lookahead):
    def pick(state):
        stage = min(state.attempt, 2) - 1
        difficulty = difficulty_bin(state.difficulty_signal, grid.shape[1])
        waits = np.array([state.standard_wait_estimate, state.specialist_wait_estimate])

        def cost(values, queue, at_cap):
            repeat, elapsed, work, unresolved = values.T
            outcome = weights[4] * (unresolved + repeat) if at_cap else weights[4] * unresolved + weights[0] * repeat
            return outcome + weights[1] * (elapsed + queue) + work * (weights[2] + weights[3] * np.array([0., 1.]))

        values = grid[stage, difficulty]
        at_cap = state.attempt == env.limit
        value = cost(values, waits, at_cap)
        if lookahead and not at_cap:
            value += values[:, 0] * cost(grid[1, difficulty], waits, state.attempt + 1 == env.limit).min()
        return ACTIONS[int(value.argmin())]

    return pick


def audit_domain(path, law, spec, config, cfg, seed):
    phases = ("ml_fit", "offline", "scale", "validation", "evaluation", "bootstrap")
    streams = dict(zip(phases, np.random.SeedSequence(seed).spawn(len(phases)), strict=True))
    logs = {}
    for phase in phases[:2]:
        folder = path / phase
        folder.mkdir(parents=True)
        generate(law, spec, config, streams[phase], folder, cfg["load"], cfg["effect"],
                 cfg["workloads"][phase], cfg["cases_per_workload"], .5)
        logs[phase] = read_replay(folder / "transitions.jsonl")
    env = LoopEnvironment(law, spec, config, cfg["effect"], scenario=True)
    workloads = {phase: scenarios(streams[phase], law, spec, cfg, cfg["workloads"][phase]) for phase in phases[2:5]}
    fifo = lambda _: "standard"
    scale = measure(env, workloads["scale"], fifo).mean(axis=0)
    weights = {"current": np.array([env.weights["repeat"], 1 / law.time_scale, 0.,
                                    env.weights["specialist_work"] / law.mean_service, env.weights["negative_exit"]])}
    weights.update({key: np.array([penalty, 1 / scale[1], .25 / scale[2], .25 / scale[2], 2.])
                    for key, penalty in cfg["rewards"].items()})
    grid = fit_grids(logs["ml_fit"], cfg, seed)
    forecasts = grid[..., [0, 2]]

    def encode(state):
        return representation(Observation(**state), forecasts, env)

    replay, events, logged = [], [], []
    for row, _ in logs["offline"]:
        x = encode(row["state"])
        done = row["done"]
        nxt = np.zeros_like(x) if done else encode(row["next_state"])
        replay.append([x, MASK, ACTIONS.index(row["action"]), 0., nxt, MASK & (not done), done, 0.])
        events.append(sum((components(event) for event in row["completed"]), np.zeros(len(FIELDS))))
        logged.append(row["reward"])
    x, masks, actions, _, nxt, next_masks, done, _ = map(np.asarray, zip(*replay))
    scaler = StandardScaler().fit(x.reshape(-1, x.shape[-1]))

    def norm(values):
        return np.clip((values - scaler.mean_) / scaler.scale_, -10., 10.).astype(np.float32)

    x, nxt = norm(x), norm(nxt)
    events = np.array(events)
    np.testing.assert_allclose(-events @ weights["current"], logged, atol=1e-9)
    policies = {"FIFO": fifo, "ML_existing": ml_rule(env, forecasts)}
    for name, weight in weights.items():
        reward = -events @ weight
        transitions = attach_returns([[None, None, None, r, None, None, d, 0.] for r, d in zip(reward, done, strict=True)])
        replay = (x, masks, actions, reward, nxt, next_masks, done, np.array([row[7] for row in transitions]))
        actor, _ = paper_rl.neural(replay, seed, cfg["cql_updates"], cfg["cql_alpha"])
        policies[f"CQL_{name}"] = lambda state, actor=actor: ACTIONS[actor(norm(representation(state, forecasts, env)), MASK)]
        for horizon in (1, 2):
            policies[f"ML_{name}_{horizon}"] = ml_plan(env, grid, weight, horizon == 2)

    limits = cfg["limits"]
    val = {key: measure(env, workloads["validation"], policy) for key, policy in policies.items()}
    ml = min((key for key in val if key.startswith("ML_") and feasible(val[key], val["FIFO"], limits)),
             default=None, key=lambda key: val[key][:, 0].mean())
    reference = ml or "FIFO"
    rl = min((key for key in val if key.startswith("CQL_") and feasible(val[key], val["FIFO"], limits, val[reference])),
             default=None, key=lambda key: val[key][:, 0].mean())
    final = {key: measure(env, workloads["evaluation"], policy) for key, policy in policies.items()}
    indices = np.random.default_rng(streams["bootstrap"]).integers(len(final[reference]), size=(cfg["bootstrap_draws"], len(final[reference])))

    def summaries(rows):
        return {key: dict(zip(FIELDS, values.mean(axis=0), strict=True)) for key, values in rows.items()}

    comparisons = {}
    for name, rows in final.items():
        diff = rows - final[reference]
        bounds = np.quantile(diff[indices].mean(axis=1), [.025, .975], axis=0)
        margins = np.column_stack((gaps(rows, final["FIFO"], limits), gaps(rows, final[reference], limits),
                                   rows[:, 3] - limits["specialist_share"] * final["FIFO"][:, 2]))
        comparisons[name] = {"feasible_on_validation": feasible(val[name], val["FIFO"], limits, val[reference]),
                             "feasible_on_evaluation": feasible(rows, final["FIFO"], limits, final[reference]),
                             "repeat_delta_95": bounds[:, 0].tolist(),
                             "all_one_sided_operating_bounds_pass": bool((np.quantile(margins[indices].mean(axis=1), .95, axis=0) <= 0).all())}
        if name.startswith("CQL_"):
            change = rows[:, 0] - final["CQL_current"][:, 0]
            comparisons[name]["repeat_vs_current_cql"] = float(change.mean())
            comparisons[name]["repeat_vs_current_cql_95"] = np.quantile(change[indices].mean(axis=1), [.025, .975]).tolist()
    passed = bool(ml and rl and comparisons[rl]["feasible_on_evaluation"] and comparisons[rl]["repeat_delta_95"][1] < 0
                  and (final[rl][:, 0] - final[ml][:, 0]).mean() <= -cfg["minimum_repeat_reduction"])
    save(path / "results.json", {"seed": seed, "weights": {k: v.tolist() for k, v in weights.items()},
                                  "decisions": len(actions), "selected_ml": ml, "selected_cql": rl,
                                  "validation": summaries(val), "evaluation": summaries(final), "comparisons": comparisons,
                                  "workload_metrics": {key: value.tolist() for key, value in final.items()},
                                  "point_limits_and_repeat_gate": passed})
    print(f"{spec['name']}: ML={ml}, CQL={rl}, passed={passed}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--config", default="configs/routing_reward_audit.json", type=Path)
    args = parser.parse_args()
    started = perf_counter()
    enable_shared_runtime()
    import torch
    torch.set_num_threads(1)
    configs = Path("configs/domain_loop_simulations.json")
    config = json.loads(configs.read_text())
    cfg = json.loads(args.config.read_text())
    seed = seeds(config["master_seed"], 1)[0]
    args.out.mkdir(parents=True)
    inputs = (args.config, configs, Path(__file__), Path(paper_rl.__file__),
              Path("src/domain_loop_simulator.py"), Path("src/domain_loop_training.py"), Path("src/benchmark_models.py"),
              Path("src/domain_loop_simulations.py"), Path("src/domain_loop_data.py"), Path("src/bpisim_data.py"))
    manifest = {"seed": seed, "settings": cfg, "python": platform.python_version(), "torch": torch.__version__,
                "device": "cpu", "hashes": {str(p): sha256(p.read_bytes()) for p in inputs}}
    save(args.out / "manifest.json", manifest)
    with threadpool_limits(limits=1):
        for index, domain, spec in selected_domains(config, "all"):
            source = acquire(spec, config["maximum_download_bytes"])
            cases, _ = load_cases(source, domain, spec)
            fit, _, _, _ = chronological_split(cases, config["fit_fraction"])
            domain_seed = seeds([seed, index, 172], 1)[0]
            audit_domain(args.out / domain, calibrate(fit, spec, config), spec, config, cfg, domain_seed)
    save(args.out / "manifest.json", manifest | {"seconds": perf_counter() - started, "status": "complete"})


if __name__ == "__main__":
    main()
