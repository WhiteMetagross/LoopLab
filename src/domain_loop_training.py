"""Short supervised, offline RL, and online RL integration runs."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from importlib.metadata import version
import json
from pathlib import Path
import platform
import sys
from time import perf_counter

import joblib
import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, mean_absolute_error

from src import paper_rl
from src.benchmark_models import FAMILIES, estimator, predict, seeds
from src.domain_loop_data import DOMAIN_CHOICES, acquire, chronological_split, load_cases, selected_domains, sha256
from src.domain_loop_simulations import arrival_minutes, generate, metrics, queue_rule, save
from src.domain_loop_simulator import ACTIONS, LoopEnvironment, Observation, calibrate, run_policy
from src.runtime import enable_shared_runtime


MASK = np.ones(len(ACTIONS), dtype=bool)


def read_replay(path, route_when_ready=False):
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    decisions, outcomes = {}, {}
    previous = None
    for row in records:
        state = row["state"]
        key = row["workload"], state["case"], state["attempt"]
        if key in decisions or not (0 < row["propensity"] < 1) or not np.isfinite(row["reward"]):
            raise ValueError("Invalid logged decision")
        decisions[key] = row
        if row["done"] != (row["next_state"] is None):
            raise ValueError("Incorrect terminal transition")
        if previous:
            if previous["workload"] == row["workload"]:
                if previous["done"] or previous["next_state"] != state:
                    raise ValueError("Broken state chain")
            elif not previous["done"]:
                raise ValueError("Truncated workload")
        next_clock = row["next_state"]["time"] if row["next_state"] else float("inf")
        for event in row["completed"]:
            result_key = row["workload"], event["case"], event["attempt"]
            if result_key in outcomes or result_key not in decisions:
                raise ValueError("Duplicate or unmatched completion")
            route_clock = event.get("service_ready", event["ready"]) if route_when_ready else event["ready"]
            if not (decisions[result_key]["action"] == event["action"]
                    and decisions[result_key]["state"]["time"] == route_clock
                    and event["ready"] <= event.get("service_ready", event["ready"]) <= event["service_start"] <= event["service_end"] <= event["end"]
                    and state["time"] <= event["end"] <= next_clock):
                raise ValueError("Completion disagrees with its decision or is not observable yet")
            outcomes[result_key] = event
        previous = row
    if not records or not records[-1]["done"] or decisions.keys() != outcomes.keys():
        raise ValueError("Replay has missing outcomes")
    return [(row, outcomes[key]) for key, row in decisions.items()]


def difficulty_bin(value, bins):
    return min(int(value * bins), bins - 1)


def ml_features(state, action, bins):
    return (min(state.attempt, 2), difficulty_bin(state.difficulty_signal, bins), int(action == "specialist"))


def prediction_rows(replay, bins):
    x, risk, work = [], [], []
    for row, result in replay:
        x.append(ml_features(Observation(**row["state"]), row["action"], bins))
        risk.append(result["outcome"] == "repeat")
        work.append(result["service_end"] - result["service_start"])
    return np.array(x), np.array(risk, dtype=int), np.array(work)


def fit_forecast(fit, holdout, cfg, seed):
    bins = cfg["difficulty_bins"]
    x, risk, work = prediction_rows(fit, bins)
    xt, rt, wt = prediction_rows(holdout, bins)
    if len(np.unique(risk)) < 2 or len(np.unique(rt)) < 2:
        raise ValueError("Prediction splits need repeat and no repeat examples")
    classifiers = {family: estimator(family, "risk", seed, cfg["trees"]).fit(x, risk) for family in FAMILIES}
    regressors = {family: estimator(family, "effort", seed, cfg["trees"]).fit(x, work) for family in FAMILIES}
    risk_scores = {}
    risks = {name: predict(model, xt, probability=True)[:, 1] for name, model in classifiers.items()}
    for name, prediction in {"prevalence": np.full(len(rt), risk.mean()), **risks}.items():
        risk_scores[name] = {"brier": float(brier_score_loss(rt, prediction)),
                             "average_precision": float(average_precision_score(rt, prediction))}
    action_means = [work[x[:, 2] == index].mean() for index in range(2)]
    work_errors = {name: float(mean_absolute_error(wt, np.maximum(0., predict(model, xt))))
                   for name, model in regressors.items()}
    report = {"fit_rows": len(x), "test_rows": len(xt), "test_repeats": int(rt.sum()), "risk": risk_scores,
              "work_mae_minutes": {"action_mean": float(mean_absolute_error(wt, np.take(action_means, xt[:, 2]))),
                                   **work_errors}}
    grid = np.array([(stage, difficulty, action) for stage in (1, 2) for difficulty in range(bins) for action in range(2)])
    forecasts = {name: np.c_[predict(classifiers[name], grid, probability=True)[:, 1],
                             np.maximum(0., predict(regressors[name], grid))].reshape(2, bins, 2, 2)
                 for name in FAMILIES}
    return forecasts, report, {"risk": classifiers, "work": regressors, "grid": forecasts}


def representation(state, forecasts, env):
    bins = forecasts.shape[1]
    stage = min(state.attempt, 2)
    difficulty = difficulty_bin(state.difficulty_signal, bins)
    values = forecasts[stage - 1, difficulty]
    risk = values[:, 0] if state.attempt < env.limit else np.zeros(2)
    service = env.law.mean_service
    waits = np.array([state.standard_wait_estimate, state.specialist_wait_estimate])
    return np.array([[stage - 1, state.difficulty_signal, np.log1p(state.case_age / env.law.time_scale),
                      min(state.standard_jobs, 10) / 10, min(state.specialist_jobs, 10) / 10,
                      action, min(waits[action] / service, 10.), risk[action], values[action, 1] / service]
                     for action in range(2)])


def ml_rule(env, forecasts):
    continuation = env.law.expected_elapsed(2, env.scenario_repeat) / env.law.time_scale + env.weights["repeat"]

    def pick(state):
        x = representation(state, forecasts, env)
        risk_difference = max(0., x[0, 7] - x[1, 7])
        delay = state.specialist_wait_estimate - state.standard_wait_estimate
        service_difference = (x[1, 8] - x[0, 8]) * env.law.mean_service
        cost = (delay + service_difference) / env.law.time_scale + env.weights["specialist_work"] * x[1, 8]
        return "specialist" if risk_difference * continuation > cost else "standard"

    return pick


def scenarios(seed, law, spec, cfg, count):
    rng = np.random.default_rng(seed)
    return [(arrival_minutes(rng, law, spec, cfg["cases_per_workload"], cfg["load"]), int(rng.integers(2**63)))
            for _ in range(count)]


def attach_returns(records):
    total = 0.
    for row in reversed(records):
        total = row[3] + (0. if row[6] else total)
        row[7] = total
    return records


def offline_arrays(replay, represent):
    records = []
    for row, _ in replay:
        state = represent(Observation(**row["state"]))
        nxt = np.zeros_like(state) if row["done"] else represent(Observation(**row["next_state"]))
        records.append([state, MASK, ACTIONS.index(row["action"]), row["reward"], nxt,
                        np.zeros_like(MASK) if row["done"] else MASK, row["done"], 0.])
    return tuple(map(np.asarray, zip(*attach_returns(records))))


def online_episode(env, workload, represent, policy):
    state = env.reset(*workload)
    records = []
    while state is not None:
        x = represent(state)
        action, log_probability = policy(x, MASK)
        state, reward, done, _ = env.step(ACTIONS[action])
        nxt = np.zeros_like(x) if done else represent(state)
        records.append([x, MASK, action, reward, nxt, np.zeros_like(MASK) if done else MASK, done, 0., log_probability])
    return attach_returns(records)


def fit_tabular(env, workloads, represent, ml_policy, cfg, seed):
    def key(state):
        x = represent(state)
        return (min(state.attempt, 2), difficulty_bin(state.difficulty_signal, 3), min(state.standard_jobs, 2),
                min(state.specialist_jobs, 2), int(x[0, 7] - x[1, 7] > .05))

    def step(action):
        state, reward, done, _ = env.step(ACTIONS[action])
        return state, reward, done

    policy, learned, updates = paper_rl.tabular(
        workloads, lambda workload: env.reset(*workload), step, key,
        lambda state: ACTIONS.index(ml_policy(state)), len(ACTIONS), seed, cfg)
    return lambda state: ACTIONS[policy(state)], learned, updates


def evaluate(env, workloads, policies):
    result = {}
    for name, policy in policies.items():
        rows, scores = [], []
        for arrivals, noise in workloads:
            cases = run_policy(env, arrivals, noise, policy)
            rows.extend(cases)
            scores.append(sum(env.reward(row) for row in env.records) / len(cases))
        result[name] = metrics(rows) | {"mean_reward_per_case": float(np.mean(scores)), "workload_rewards": scores}
    return result


def run_domain(law, spec, cfg, train, seed, path, torch):
    splits = np.random.SeedSequence(seed).spawn(5)
    replays = {}
    for split, child, count in zip(("ml_fit", "ml_test", "offline"), splits[:3],
                                    (train["ml_fit_workloads"], train["ml_test_workloads"], train["offline_workloads"])):
        target = path / split
        target.mkdir()
        generate(law, spec, cfg, child, target, train["load"], train["effect"], count,
                 train["cases_per_workload"], train["behavior_probability"])
        replays[split] = read_replay(target / "transitions.jsonl")
    forecasts, prediction, models = fit_forecast(replays["ml_fit"], replays["ml_test"], train, seed)
    env = LoopEnvironment(law, spec, cfg, train["effect"], scenario=True)
    ml_policies = {f"{name}_ML": ml_rule(env, grid) for name, grid in forecasts.items()}

    def represent(state):
        return representation(state, forecasts["CatBoost"], env)

    replay = offline_arrays(replays["offline"], represent)
    np.savez_compressed(path / "offline_replay.npz", **dict(zip(("x", "mask", "action", "reward", "next_x", "next_mask", "done", "returns"), replay)))
    fitted, models["fitted_q"] = paper_rl.fitted(replay, seed, train["fitted_q_rounds"], train["trees"])
    cql, cql_state = paper_rl.neural(replay, seed, train["cql_updates"], train["cql_alpha"])
    online = scenarios(splits[3], law, spec, train, train["online_workloads"])
    q, models["tabular_q"], q_updates = fit_tabular(env, online, represent, ml_policies["CatBoost_ML"], train, seed)

    def ppo_workloads(policy):
        for workload in online:
            yield online_episode(env, workload, represent, policy)

    ppo, ppo_state, ppo_counts = paper_rl.ppo(ppo_workloads, replay[0].shape[-1], seed, train["ppo_horizon"])
    joblib.dump(models, path / "models.joblib")
    torch.save(cql_state, path / "cql.pt")
    torch.save(ppo_state, path / "ppo.pt")
    fitted_model = joblib.load(path / "models.joblib")["fitted_q"]
    assert np.allclose(fitted_model.predict(replay[0][0]), models["fitted_q"].predict(replay[0][0]))
    for name, expected in (("cql", cql_state), ("ppo", ppo_state)):
        restored = torch.load(path / f"{name}.pt", map_location="cpu", weights_only=True)
        assert all(torch.equal(expected[key], restored[key]) for key in expected)

    holdout = scenarios(splits[4], law, spec, train, train["policy_test_workloads"])
    policies = {"standard_only": lambda _: "standard", "queue_aware": queue_rule(env), **ml_policies,
                "ml_tabular_q": q, "ml_fitted_q": lambda state: ACTIONS[fitted(represent(state), MASK)],
                "ml_cql": lambda state: ACTIONS[cql(represent(state), MASK)],
                "ml_ppo": lambda state: ACTIONS[ppo(represent(state), MASK)]}
    policy_metrics = evaluate(env, holdout, policies)
    policy_metrics["ml_only"] = policy_metrics["CatBoost_ML"]
    return {"prediction": prediction, "offline_actions": dict(Counter(row["action"] for row, _ in replays["offline"])),
            "tabular_q_updates": q_updates, "cql_updates": train["cql_updates"], "ppo": ppo_counts,
            "policy_metrics": policy_metrics, "checkpoint_reload_verified": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/domain_loop_simulations.json"))
    parser.add_argument("--training-config", type=Path, default=Path("configs/domain_loop_training_smoke.json"))
    parser.add_argument("--domain", type=str.lower, choices=DOMAIN_CHOICES, default="both")
    parser.add_argument("--seed-index", type=int, default=0)
    args = parser.parse_args()
    if not 0 <= args.seed_index < 10000:
        parser.error("Seed index must be in [0, 10000)")
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    train = json.loads(args.training_config.read_text(encoding="utf-8"))
    enable_shared_runtime(train["shared_readonly_runtime_env"])
    import torch
    torch.set_num_threads(1)
    start = perf_counter()
    seed = seeds(cfg["master_seed"], args.seed_index + 1)[-1]
    args.out.mkdir(parents=True, exist_ok=False)
    manifest = {"seed": seed, "seed_index": args.seed_index, "config": cfg, "training_config": train,
                "domain": args.domain, "runtime": {"device": "CPU", "python": sys.version.split()[0], "torch": torch.__version__,
                "torch_compiled_cuda": torch.version.cuda, "platform": platform.platform(),
                **{name: version(name) for name in ("numpy", "scikit-learn", "catboost", "xgboost-cpu", "joblib")}},
                "code_sha256": {file.name: sha256(file.read_bytes()) for file in
                                (*Path(__file__).parent.glob("domain_loop*.py"), Path(paper_rl.__file__),
                                 *(Path(__file__).with_name(name) for name in
                                   ("benchmark_models.py", "bpisim_data.py", "bpi_queue.py", "cycle_data.py",
                                    "loop_feasibility.py")))}}
    save(args.out / "manifest.json", manifest)
    for index, domain, spec in selected_domains(cfg, args.domain):
        source = acquire(spec, cfg["maximum_download_bytes"])
        cases, _ = load_cases(source, domain, spec)
        fit, _, _, _ = chronological_split(cases, cfg["fit_fraction"])
        law = calibrate(fit, spec, cfg)
        target = args.out / domain
        target.mkdir()
        domain_seed = int(np.random.SeedSequence([seed, index]).generate_state(1)[0] % (2**31 - 1))
        report = run_domain(law, spec, cfg, train, domain_seed, target, torch)
        name = spec.get("name", domain)
        save(target / "results.json", report | {"seed": domain_seed, "simulator": name})
        print(f"{name}: {report['prediction']['fit_rows']} supervised fit rows, {sum(report['offline_actions'].values())} offline decisions, {report['tabular_q_updates']} Q updates", flush=True)
    save(args.out / "manifest.json", manifest | {"status": "complete", "elapsed_seconds": perf_counter() - start})


if __name__ == "__main__":
    main()
