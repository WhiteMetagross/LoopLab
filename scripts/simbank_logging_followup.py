"""Train under two logging policies and audit off-policy evaluation against native replay."""

from fcntl import LOCK_EX, flock
from functools import lru_cache
from itertools import combinations
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import StandardScaler

from scripts.check_negotiation_randomness import IndependentNegotiationEnvironment
from scripts.simbank_timed_override import plan
from scripts.simfactory_tradeoff_replay import read
from src import paper_rl
from src.benchmark_models import estimator, predict
from src.domain_loop_data import sha256
from src.domain_loop_simulations import save
from src.negotiation_support import bank_rate, run_policy
from src.paper_simbank import FIELDS, state_record
from src.runtime import enable_shared_runtime


def log_cases(engine, cases, rates, bias, seed):
    rng = np.random.default_rng(seed)
    logs = []
    for case in cases:
        propensities = []

        def actor(state):
            probabilities = np.full(len(rates), (1 - bias) / len(rates))
            probabilities[bank_rate(state, rates)] += bias
            action = int(rng.choice(len(rates), p=probabilities))
            propensities.append(float(probabilities[action]))
            return action

        outcome, trace = run_policy(engine, case, actor)
        assert len(trace) == len(propensities)
        for row, probability in zip(trace, propensities, strict=True):
            row["propensity"] = probability
        logs.append((outcome, trace))
    return logs


def outcomes(row):
    return [max(0, row["refusals"] - 1) if field == "repeat_refusals" else row[field] for field in FIELDS]


def fit(logs, rates, seed, trees, refusal_cost, cancellation_cost, scale, family="CatBoost"):
    trace = [(outcome, row) for outcome, episode in logs for row in episode]
    states = [row["state"] for _, row in trace]
    actions = np.array([row["action"] for _, row in trace])
    vector = DictVectorizer(sparse=False).fit([state_record(s, rates[a]) for s, a in zip(states, actions, strict=True)])

    def matrix(prefixes):
        return vector.transform([state_record(s, rate) for s in prefixes for rate in rates])

    x_all = matrix(states).reshape(len(states), len(rates), -1)
    x = x_all[np.arange(len(actions)), actions]
    legal = np.array([rates[a] >= s["minimum_rate"] - 1e-9 for s, a in zip(states, actions, strict=True)])
    accepted = np.array([row["accepted_now"] for _, row in trace], bool)
    refused = np.array([row["immediate_refusal"] for _, row in trace], bool)
    if not legal.any() or not accepted.any() or len(set(refused[legal])) != 2:
        raise ValueError("Logging replay lacks refusal or acceptance outcomes")
    risk = estimator(family, "risk", seed, trees).fit(x[legal], refused[legal].astype(int))
    profit = estimator(family, "profit", seed, trees).fit(x[accepted], [row["terminal_profit"] for _, row in trace if row["accepted_now"]])

    def predictions(z):
        return predict(risk, z, True)[:, 1], predict(profit, z)

    def enrich(z):
        p, v = predictions(z.reshape(-1, z.shape[-1]))
        return np.column_stack((z.reshape(-1, z.shape[-1]), p, v / scale)).reshape(*z.shape[:-1], -1).astype(np.float32)

    encoded = enrich(x_all)
    done = np.array([row["next_state"] is None for _, row in trace])
    next_raw = np.zeros_like(encoded)
    next_raw[~done] = enrich(matrix([row["next_state"] for _, row in trace if row["next_state"] is not None]).reshape(-1, len(rates), x.shape[-1]))
    reward = np.array([(outcome["profit"] - cancellation_cost * outcome["cancelled"] if terminal else 0.)
                       for terminal, (outcome, _) in zip(done, trace, strict=True)])
    reward = (reward - refusal_cost * refused) / scale
    scaler = StandardScaler().fit(encoded.reshape(-1, encoded.shape[-1]))

    def normalize(z):
        return np.clip((z - scaler.mean_) / scaler.scale_, -10, 10).astype(np.float32)

    def forecast(prefixes):
        return tuple(v.reshape(len(prefixes), len(rates)) for v in predictions(matrix(prefixes)))

    def observe(items):
        return normalize(enrich(matrix([dict(items)])))

    masks = np.ones(x_all.shape[:2], bool)
    return (normalize(encoded), masks, actions, reward, normalize(next_raw), masks & ~done[:, None], done, reward), forecast, observe, (vector, risk, profit, scaler)


def support(logs, horizon, actions):
    counts = np.zeros((horizon, actions), dtype=int)
    for _, trace in logs:
        for row in trace:
            counts[row["state"]["offer_count"], row["action"]] += 1
    return counts.tolist()


def estimate(logs, policy, minimum_ess):
    rows, weights, depths = [], [], []
    for outcome, trace in logs:
        log_weight = 0.
        for row in trace:
            if policy(tuple(row["state"].items())) != row["action"]:
                log_weight = -np.inf
                break
            log_weight -= np.log(row["propensity"])
        rows.append(outcomes(outcome))
        weights.append(np.exp(log_weight))
        depths.append(len(trace))
    weights, depths, rows = np.asarray(weights), np.asarray(depths), np.asarray(rows)

    def ess(w):
        return float(w.sum() ** 2 / np.dot(w, w)) if w.any() else 0.

    action_ess = ess(weights[depths > 0])
    supported = action_ess >= minimum_ess
    report = {"status": "supported" if supported else "insufficient_overlap", "ess": ess(weights),
              "ess_with_decisions": action_ess, "ess_multiple_decisions": ess(weights[depths > 1]),
              "matching_with_decisions": int(((weights > 0) & (depths > 0)).sum()),
              "max_weight": float(weights.max()), "estimate": None}
    if supported:
        report["estimate"] = dict(zip(FIELDS, np.average(rows, axis=0, weights=weights).tolist(), strict=True))
    return report, np.column_stack((weights, depths))


def run(folder, index, seed, config, profile, progress, device):
    enable_shared_runtime()
    import torch

    torch.set_num_threads(1)
    paper_rl.DEVICE = device
    spec = config["bank"]
    paper = read(Path(config["paper_config"]))
    cfg = read(Path(paper["simbank_config"])) | {"seed": config["master_seed"]}
    rates, horizon, scale = cfg["rates"], cfg["max_offers"], cfg["profit_scale"]
    refusal, cancellation = (paper["simbank"][key] for key in ("refusal_penalty", "cancellation_penalty"))
    engine = IndependentNegotiationEnvironment(cfg)
    start = spec["scenario_base"] + (index + profile["bank_seed_offset"]) * spec["seed_stride"]
    train = range(start, start + profile["bank_fit"])
    test_start = start + spec["test_offset"]
    test = range(test_start, test_start + profile["bank_cases"])
    if train.stop > test.start or test.stop > start + spec["seed_stride"]:
        raise ValueError("SimBank scenario ranges overlap")
    behavior = {"random": 0., "biased": spec["bias"]}
    policies = {"Bank": lambda items: bank_rate(dict(items), rates)}
    training_support = {}
    for label, bias in behavior.items():
        progress(f"fit_{label}_replay")
        logs = log_cases(engine, train, rates, bias, np.random.SeedSequence([seed, profile["namespace"], 0]))
        training_support[label] = support(logs, horizon, len(rates))
        data, forecast, observe, fitted = fit(logs, rates, seed, profile["bank_trees"], refusal, cancellation,
                                              scale, spec.get("forecast_family", "CatBoost"))
        joblib.dump(fitted, folder / f"{label}_predictors.joblib")
        save(folder / f"{label}_train.json", {"range": [train.start, train.stop], "bias": bias,
                                             "support": training_support[label], "transitions": len(data[2])})
        np.savez_compressed(folder / f"{label}_train.npz", action=data[2], reward=data[3], done=data[6],
                            propensity=np.array([row["propensity"] for _, trace in logs for row in trace]))

        def ml(items, forecast=forecast):
            return int(plan([dict(items)], rates, horizon, forecast, refusal, cancellation).argmax())

        policies[f"ML_{label}"] = lru_cache(maxsize=24576)(ml)
        with (folder.parent / "gpu.lock").open("a") as lock:
            if device == "cuda":
                progress(f"waiting_for_gpu_{label}")
                flock(lock, LOCK_EX)
            progress(f"fit_{label}_cql")
            actor, weights = paper_rl.neural(data, seed, profile["bank_updates"], spec["alpha"])
            torch.save(weights, folder / f"{label}_cql.pt")
            if device == "cuda":
                torch.cuda.empty_cache()

        def cql(items, actor=actor, observe=observe):
            return actor(observe(items), np.ones(len(rates), bool))

        policies[f"CQL_{label}"] = lru_cache(maxsize=24576)(cql)

    progress("native_evaluation")
    rows = {name: np.empty((len(test), len(FIELDS))) for name in policies}
    for index, case in enumerate(test):
        for name, policy in policies.items():
            state = engine.reset(case)
            while state is not None:
                state = engine.step(policy(tuple(state.items())))
            row = engine.result()
            if row["accepted"] + row["cancelled"] != 1 or not row["refusals"] <= row["offers"] <= horizon:
                raise ValueError("Invalid native case result")
            rows[name][index] = outcomes(row)
        if (index + 1) % 1000 == 0:
            progress(f"native_evaluation_{index + 1}")

    evaluation = {}
    native = {name: dict(zip(FIELDS, values.mean(axis=0).tolist(), strict=True)) for name, values in rows.items()}
    for label, bias in behavior.items():
        progress(f"logged_evaluation_{label}")
        logs = log_cases(engine, test, rates, bias, np.random.SeedSequence([seed, profile["namespace"], 1]))
        reports, weights = {}, {}
        for name, policy in policies.items():
            reports[name], weights[name] = estimate(logs, policy, spec["minimum_ess"])
        logged = {"outcomes": np.asarray([outcomes(row) for row, _ in logs])}
        actions = np.full((len(logs), horizon), -1, dtype=int)
        propensity = np.zeros((len(logs), horizon))
        for i, (_, trace) in enumerate(logs):
            for j, row in enumerate(trace):
                actions[i, j], propensity[i, j] = row["action"], row["propensity"]
        np.savez_compressed(folder / f"{label}_logged.npz", **logged, actions=actions, propensities=propensity, **weights)
        eligible = [name for name, entry in reports.items() if entry["estimate"] is not None]
        for name in eligible:
            estimate_ = reports[name]["estimate"]
            reports[name]["estimate_minus_native"] = {k: estimate_[k] - native[name][k] for k in FIELDS}
            if "Bank" in eligible:
                reports[name]["gain_error_vs_bank"] = {k: reports[name]["estimate_minus_native"][k] - (
                    reports["Bank"]["estimate"][k] - native["Bank"][k]) for k in FIELDS}
        ranking = {}
        for metric in ("profit", "repeat_refusals"):
            pairs = [(a, b) for a, b in combinations(eligible, 2) if native[a][metric] != native[b][metric]]
            ranking[metric] = {"comparable_pairs": len(pairs), "strict_reversals": sum(
                (native[a][metric] - native[b][metric]) * (reports[a]["estimate"][metric] - reports[b]["estimate"][metric]) < 0
                for a, b in pairs)}
        evaluation[label] = {"policies": reports, "ranking": ranking, "support": support(logs, horizon, len(rates))}

    contrasts = {"ML_biased_minus_random": {"ML_biased": 1, "ML_random": -1},
                 "CQL_biased_minus_random": {"CQL_biased": 1, "CQL_random": -1},
                 "CQL_minus_ML_random": {"CQL_random": 1, "ML_random": -1},
                 "CQL_minus_ML_biased": {"CQL_biased": 1, "ML_biased": -1},
                 "interaction": {"CQL_biased": 1, "ML_biased": -1, "CQL_random": -1, "ML_random": 1}}
    raw = Path(cfg["parameter_file"])
    save(folder / "design.json", {"fields": FIELDS, "contrasts": contrasts, "seed": seed, "settings": spec,
                                  "forecast_family": spec.get("forecast_family", "CatBoost"),
                                  "data": {str(raw): sha256(raw.read_bytes())}, "train_range": [train.start, train.stop],
                                  "case_range": [test.start, test.stop], "training_support": training_support, "ope": evaluation})
    np.savez_compressed(folder / "paired.npz", **rows)
