"""Calibrated refusal/effort prediction, finite horizon Q and ML planning."""
from __future__ import annotations

import argparse
from copy import deepcopy
import importlib.metadata
import json
from pathlib import Path
import platform
import signal
import time

from catboost import CatBoostClassifier, CatBoostRegressor
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from src.loop_feasibility import digest, write_json
from src.negotiation_support import (FEATURES, NegotiationEnvironment, bank_rate, cancel_profit,
                                     refusal_successor, run_policy)


def options(config):
    return dict(**config["catboost"], random_seed=config["seed"], thread_count=1,
                verbose=False, allow_writing_files=False)


def action_rows(states, rates):
    return pd.DataFrame([{**state, "rate": rate, "rate_gap": rate - state["minimum_rate"]}
                         for state in states for rate in rates])[FEATURES + ["rate", "rate_gap"]]


def frame_rows(transitions, config):
    return pd.DataFrame([{**row["state"], "rate": config["rates"][row["action"]],
                          "rate_gap": config["rates"][row["action"]] - row["state"]["minimum_rate"]}
                         for row in transitions])[FEATURES + ["rate", "rate_gap"]]


def collect(env, config, part, count, rng):
    rows, cases = [], []
    for i in range(count):
        scenario = config["seed"] + config["scenario_offsets"][part] + i
        result, transitions = run_policy(env, scenario, lambda state: int(rng.integers(len(config["rates"]))))
        cases.append({"scenario": scenario, **result})
        rows.extend({"scenario": scenario, **row} for row in transitions)
    return rows, cases


def raw_probability(bundle, X):
    return bundle["risk"].predict_proba(X)[:, 1]


def calibrate_input(probability):
    p = np.clip(probability, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))[:, None]


def probability(bundle, X):
    return bundle["platt"].predict_proba(calibrate_input(raw_probability(bundle, X)))[:, 1]


def augmented(bundle, X, config):
    result = X.copy()
    result["ML_refusal"] = probability(bundle, X)
    result["ML_remaining_effort"] = bundle["effort"].predict(X) / 1440
    result["ML_remaining_refusals"] = bundle["count"].predict(X)
    result["ML_behavior_profit"] = bundle["profit"].predict(X) / config["profit_scale"]
    return result


def prediction_metrics(bundle, rows, config):
    X = frame_rows(rows, config)
    y = np.array([row["immediate_refusal"] for row in rows])
    probabilities = {"Stage_action_lookup": np.array([bundle["lookup_risk"].get((row["state"]["offer_count"], row["action"]), bundle["global_risk"]) for row in rows]),
                     "Logistic_regression": bundle["LR"].predict_proba(bundle["scaler"].transform(bundle["vector"].transform(X.to_dict("records"))))[:, 1],
                     "CatBoost_raw": raw_probability(bundle, X), "CatBoost_calibrated": probability(bundle, X)}
    result = {"opportunities": len(rows), "cases": len({r["scenario"] for r in rows}), "risk": {}, "effort": {}, "count": {}}
    for name, p in probabilities.items():
        result["risk"][name] = {"Brier": float(np.mean((p - y) ** 2)), "predicted_rate": float(p.mean()), "observed_rate": float(y.mean())}
    for target, modelkey, lookup in [("remaining_effort", "effort", "lookup_effort"), ("remaining_refusals", "count", "lookup_count")]:
        y = np.array([r[target] for r in rows])
        estimates = {"Stage_action_lookup": np.array([bundle[lookup].get((r["state"]["offer_count"], r["action"]), bundle["global_" + modelkey]) for r in rows]),
                     "CatBoost": bundle[modelkey].predict(X)}
        result[modelkey] = {name: {"RMSE": float(np.sqrt(np.mean((p - y) ** 2))), "MAE": float(np.mean(abs(p - y)))} for name, p in estimates.items()}
    return result


def fit_stage(config, out):
    started = time.perf_counter()
    env, rng = NegotiationEnvironment(config), np.random.default_rng(config["seed"])
    data = {}
    for part, key in [("fit", "fit_cases"), ("q_fit", "q_fit_cases"), ("calibration", "calibration_cases")]:
        rows, cases = collect(env, config, part, config[key], rng)
        data[part] = rows
        pd.DataFrame(cases).to_csv(out / f"{part}_cases.csv", index=False)
        print(f"Generated {part}: {len(cases)} cases, {len(rows)} opportunities.", flush=True)
    groups = {part: set(range(config["seed"] + config["scenario_offsets"][part], config["seed"] + config["scenario_offsets"][part] + config[key]))
              for part, key in [("fit", "fit_cases"), ("q_fit", "q_fit_cases"), ("calibration", "calibration_cases")]}
    assert all(not values & groups[other] for part, values in groups.items() for other in groups if part != other)
    rows, cal = data["fit"], data["calibration"]
    X, XC = frame_rows(rows, config), frame_rows(cal, config)
    opts = options(config)
    bundle = {"risk": CatBoostClassifier(**opts, loss_function="Logloss").fit(X, [r["immediate_refusal"] for r in rows], cat_features=["last_activity"])}
    for key, target in [("effort", "remaining_effort"), ("count", "remaining_refusals"), ("profit", "terminal_profit")]:
        bundle[key] = CatBoostRegressor(**opts, loss_function="RMSE").fit(X, [r[target] for r in rows], cat_features=["last_activity"])
    accepted = [r for r in rows if r["accepted_now"]]
    bundle["accepted_profit"] = CatBoostRegressor(**opts, loss_function="RMSE").fit(frame_rows(accepted, config), [r["terminal_profit"] for r in accepted], cat_features=["last_activity"])
    bundle["platt"] = LogisticRegression(C=1., random_state=config["seed"]).fit(calibrate_input(raw_probability(bundle, XC)), [r["immediate_refusal"] for r in cal])
    vector = DictVectorizer(sparse=False)
    Z = vector.fit_transform(X.to_dict("records"))
    scaler = StandardScaler().fit(Z)
    lr = LogisticRegression(C=1., max_iter=500, random_state=config["seed"]).fit(scaler.transform(Z), [r["immediate_refusal"] for r in rows])
    bundle.update(vector=vector, scaler=scaler, LR=lr)
    for key, target in [("risk", "immediate_refusal"), ("effort", "remaining_effort"), ("count", "remaining_refusals")]:
        values = {}
        for r in rows:
            values.setdefault((r["state"]["offer_count"], r["action"]), []).append(r[target])
        bundle["lookup_" + key] = {k: float(np.mean(v)) for k, v in values.items()}
        bundle["global_" + key] = float(np.mean([r[target] for r in rows]))
    pooled = rows + data["q_fit"]
    bundle["pooled_profit"] = CatBoostRegressor(**opts, loss_function="RMSE").fit(frame_rows(pooled, config), [r["terminal_profit"] for r in pooled], cat_features=["last_activity"])
    joblib.dump(bundle, out / "models.joblib", compress=3)
    joblib.dump(data, out / "transitions.joblib", compress=3)
    report = {"rows": {part: len(rows) for part, rows in data.items()}, "calibration": prediction_metrics(bundle, cal, config),
              "accepted_fit_rows": len(accepted), "group_overlap": 0, "seconds": time.perf_counter() - started}
    write_json(out / "fit.json", report)
    print(json.dumps(report, indent=2), flush=True)


class FittedQ:
    def __init__(self, models, bundle, config, use_ml=False):
        self.models, self.bundle, self.config, self.use_ml = models, bundle, config, use_ml

    def values(self, states):
        rows = action_rows(states, self.config["rates"])
        if self.use_ml:
            rows = augmented(self.bundle, rows, self.config)
        result = np.empty((len(states), len(self.config["rates"])))
        stages = np.array([s["offer_count"] for s in states])
        for stage in np.unique(stages):
            selected = np.flatnonzero(stages == stage)
            indices = np.concatenate([np.arange(i * len(self.config["rates"]), (i + 1) * len(self.config["rates"])) for i in selected])
            result[selected] = self.models[int(stage)].predict(rows.iloc[indices]).reshape(len(selected), -1)
        return result

    def act(self, state):
        return int(np.argmax(self.values([state])[0]))


def q_stage(config, out):
    started = time.perf_counter()
    data = joblib.load(out / "transitions.joblib")
    bundle = joblib.load(out / "models.joblib")
    rows = data["q_fit"]
    output, metadata = {}, {}
    for label, use_ml, monte_carlo in [("FQI_plain", False, False), ("FQI_ML", True, False), ("Matched_MC", False, True)]:
        models, stages = {}, {}
        for stage in reversed(range(config["max_offers"])):
            rr = [r for r in rows if r["state"]["offer_count"] == stage]
            if not rr:
                raise ValueError("No transition support at offer stage.")
            y = np.array([r["terminal_profit"] / config["profit_scale"] for r in rr])
            if not monte_carlo:
                mask = np.array([r["next_state"] is not None for r in rr])
                if mask.any():
                    critic = FittedQ(models, bundle, config, use_ml)
                    next_states = [r["next_state"] for r in rr if r["next_state"] is not None]
                    y[mask] = critic.values(next_states).max(axis=1)
            X = frame_rows(rr, config)
            if use_ml:
                X = augmented(bundle, X, config)
            model = CatBoostRegressor(**options(config), loss_function="RMSE")
            model.fit(X, y, cat_features=["last_activity"])
            models[stage] = model
            stages[str(stage)] = {"rows": len(rr), "target_mean": float(y.mean())}
        output[label] = models
        metadata[label] = stages
    joblib.dump(output, out / "critics.joblib", compress=3)
    write_json(out / "q_fit.json", {"stages": metadata, "seconds": time.perf_counter() - started})
    print(json.dumps({"stages": metadata, "seconds": time.perf_counter() - started}, indent=2), flush=True)


def native_acceptance(state, rate):
    amount, elapsed = state["amount"], state["offer_time"]
    a = .9999 if amount > 80000 else (amount - 1000) / 79000
    interest = (rate - .069) / .022
    p = .5 + (-2 * a * interest + 1) * .25 + (-2 * (elapsed - 4.9) / 8.1 + 1) * .25
    p = p * .6 if p < .5 else .7 + (p - .5) * .6
    if rate >= .12 or elapsed >= 12:
        p = 0.
    return float(np.clip(p, .0001, .9999))


def estimated_accepted_profit(state, rate):
    discount = .03 + (10 - state["est_quality"]) / 200
    return state["amount"] * ((1 + rate) / (1 + discount)) ** 10 - state["amount"] - state["cum_cost"] - 400 - 10 - 100


class Planner:
    def __init__(self, bundle, config, learned=True):
        self.bundle, self.config, self.learned = bundle, config, learned

    def values(self, states):
        rows = action_rows(states, self.config["rates"])
        if self.learned:
            refusal = probability(self.bundle, rows).reshape(len(states), -1)
            accepted_profit = self.bundle["accepted_profit"].predict(rows).reshape(len(states), -1)
        else:
            refusal = np.array([[1 - native_acceptance(s, a) for a in self.config["rates"]] for s in states])
            accepted_profit = np.array([[estimated_accepted_profit(s, a) for a in self.config["rates"]] for s in states])
        q = np.empty_like(refusal)
        next_states, locations = [], []
        for i, state in enumerate(states):
            for j, rate in enumerate(self.config["rates"]):
                if rate < state["minimum_rate"] - 1e-9:
                    q[i, j] = cancel_profit(state)
                    continue
                q[i, j] = (1 - refusal[i, j]) * accepted_profit[i, j]
                if state["refusal_count"] + 1 >= self.config["max_offers"]:
                    q[i, j] += refusal[i, j] * cancel_profit(state, refusal=True)
                else:
                    next_states.append(refusal_successor(state, rate))
                    locations.append((i, j))
        if next_states:
            continuation = self.values(next_states).max(axis=1)
            for (i, j), value in zip(locations, continuation):
                q[i, j] += refusal[i, j] * value
        return q

    def act(self, state):
        return int(np.argmax(self.values([state])[0]))


def policy_summary(frame):
    return {key: float(frame[key].mean()) for key in ["profit", "accepted", "cancelled", "offers", "refusals", "revised_offers", "negotiation_effort", "elapsed_days", "native_cost"]}


def evaluate_policies(env, config, seeds, policies, part):
    records = []
    for label, policy in policies.items():
        for scenario in seeds:
            result, _ = run_policy(env, scenario, policy)
            records.append({"part": part, "policy": label, "scenario": scenario, **result})
        print(f"Evaluated {part} {label}.", flush=True)
    return pd.DataFrame(records)


def policies(config, bundle, critics):
    def model_act(model, state):
        return int(np.argmax(model.predict(action_rows([state], config["rates"]))))
    result = {"Bank": lambda state: bank_rate(state, config["rates"])}
    result.update({f"Fixed_{rate:g}": lambda state, i=i: i for i, rate in enumerate(config["rates"])})
    result["Greedy_MC"] = lambda state: model_act(bundle["profit"], state)
    result["Pooled_MC"] = lambda state: model_act(bundle["pooled_profit"], state)
    for label in critics:
        result[label] = FittedQ(critics[label], bundle, config, label == "FQI_ML").act
    result["ML_DP"] = Planner(bundle, config, True).act
    result["Mechanism_DP_reference"] = Planner(bundle, config, False).act
    return result


def evaluate_stage(config, out):
    started = time.perf_counter()
    env = NegotiationEnvironment(config)
    bundle = joblib.load(out / "models.joblib")
    critics = joblib.load(out / "critics.joblib")
    candidates = policies(config, bundle, critics)
    groups = {}
    for part, count in [("policy_validation", config["policy_validation_cases"]), ("development", config["development_cases"])]:
        seeds = list(range(config["seed"] + config["scenario_offsets"][part], config["seed"] + config["scenario_offsets"][part] + count))
        frame = evaluate_policies(env, config, seeds, candidates, part)
        frame.to_csv(out / f"{part}_policies.csv", index=False)
        groups[part] = {name: policy_summary(group) for name, group in frame.groupby("policy")}
        if part == "policy_validation":
            best_simple = max(["Bank"] + [p for p in candidates if p.startswith("Fixed_")], key=lambda p: groups[part][p]["profit"])
            best_ml = max(["Greedy_MC", "Pooled_MC", "Matched_MC", "ML_DP"], key=lambda p: groups[part][p]["profit"])
            best_rl = max(["FQI_plain", "FQI_ML"], key=lambda p: groups[part][p]["profit"])
            write_json(out / "selection_before_development.json", {"simple": best_simple, "ML": best_ml, "RL": best_rl, "validation": groups[part]})
    rng = np.random.default_rng(config["seed"])
    prediction_rows, _ = collect(env, config, "development", config["development_cases"], rng)
    joblib.dump(prediction_rows, out / "development_prediction_rows.joblib", compress=3)
    prediction = prediction_metrics(bundle, prediction_rows, config)
    table = pd.read_csv(out / "development_policies.csv").pivot(index="scenario", columns="policy", values="profit")
    paired = {}
    for candidate in ["FQI_ML", "FQI_plain", "ML_DP", "Pooled_MC", "Matched_MC"]:
        paired[candidate] = {}
        for baseline in sorted({best_simple, best_ml, "Bank", "Greedy_MC"}):
            d = table[candidate] - table[baseline]
            se = d.std(ddof=1) / np.sqrt(len(d))
            paired[candidate][baseline] = {"mean": float(d.mean()), "normal95_lower": float(d.mean() - 1.96 * se), "normal95_upper": float(d.mean() + 1.96 * se)}
    dev, verdicts = groups["development"], {}
    for label in ["FQI_plain", "FQI_ML", "ML_DP", "Pooled_MC", "Matched_MC"]:
        gain = 100 * (dev[label]["profit"] / dev[best_simple]["profit"] - 1)
        cancel_change = 100 * (dev[label]["cancelled"] - dev["Bank"]["cancelled"])
        ml_gain = 100 * (dev[label]["profit"] / max(dev[p]["profit"] for p in ["Greedy_MC", "Pooled_MC", "Matched_MC", "ML_DP"]) - 1)
        verdicts[label] = {"profit_gain_percent_vs_selected_simple": gain, "cancellation_change_points_vs_bank": cancel_change,
                           "gain_percent_vs_strongest_ML": ml_gain,
                           "policy_gate": bool(gain >= config["policy_profit_gain_gate_percent"] and cancel_change <= config["cancellation_tolerance_percentage_points"]),
                           "superior_RL_gate": bool(label.startswith("FQI") and ml_gain >= config["RL_gain_over_strong_ML_percent"] and cancel_change <= config["cancellation_tolerance_percentage_points"])}
    write_json(out / "evaluation.json", {"prediction": prediction, "policy": groups, "paired": paired, "verdicts": verdicts,
                                         "selection": {"simple": best_simple, "ML": best_ml, "RL": best_rl}, "seconds": time.perf_counter() - started,
                                         "single_seed_only": True, "protected_final_scored": False})
    print(json.dumps({"prediction": prediction, "selection": {"simple": best_simple, "ML": best_ml, "RL": best_rl},
                      "development": dev, "verdicts": verdicts, "seconds": time.perf_counter() - started}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["fit", "q", "evaluate"], required=True)
    parser.add_argument("--config", default="configs/negotiation_pilot.json")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    out = Path(config["out"])
    out.mkdir(parents=True, exist_ok=True)
    targets = {"fit": "fit.json", "q": "q_fit.json", "evaluate": "evaluation.json"}
    if (out / targets[args.stage]).exists():
        raise FileExistsError("Completed stage exists.")
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("Negotiation stage cap reached.")))
    signal.alarm(config["stage_wall_seconds"])
    native = ["external/SimBank/SimBank/activity_execution.py", "external/SimBank/SimBank/extra_flow_conditions.py", "external/SimBank/SimBank/simulation.py"]
    before = {path: digest(path) for path in native}
    with threadpool_limits(limits=1):
        {"fit": fit_stage, "q": q_stage, "evaluate": evaluate_stage}[args.stage](config, out)
    assert before == {path: digest(path) for path in native}
    write_json(out / f"{args.stage}_provenance.json", {"seed": config["seed"], "config": config, "stage": args.stage,
               "native_hashes": before, "native_modified": False, "source_sha256": {p: digest(p) for p in [args.config, config["parameter_file"], __file__, "src/negotiation_support.py"]},
               "python": platform.python_version(), "platform": platform.platform(), "device": "CPU", "threads": 1,
               "versions": {p: importlib.metadata.version(p) for p in ["numpy", "pandas", "catboost", "scikit-learn", "simpy", "pm4py"]}})
    signal.alarm(0)


if __name__ == "__main__":
    main()
