"""Full SimBank comparison on fresh independent negotiation workloads."""

from functools import lru_cache
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

from scripts.check_negotiation_randomness import IndependentNegotiationEnvironment
from scripts.simbank_constrained_reward import select_low_refusal
from scripts.simbank_timed_override import plan
from src.benchmark_models import FAMILIES, estimator, predict
from src.domain_loop_simulations import save
from src.loop_feasibility import digest
from src.negotiation_pilot import calibrate_input, estimated_accepted_profit, native_acceptance
from src.negotiation_support import FEATURES, bank_rate, run_policy
from src.paper_utils import scores, targets
from src.simulator_benchmark import fit_policies


FIELDS = ("profit", "refusals", "repeat_refusals", "negotiation_effort", "cancelled", "accepted", "offers", "revised_offers", "elapsed_days", "native_cost")
TARGETS = {"risk": "immediate_refusal", "effort": "remaining_effort", "count": "remaining_refusals", "profit": "terminal_profit"}


def state_record(state, rate):
    return {key: state[key] for key in FEATURES} | {"rate": rate, "rate_gap": rate - state["minimum_rate"]}


class Negotiation:
    def __init__(self, engine, scenario, config, observe, refusal_cost, cancel_cost):
        self.env, self.state = engine, engine.reset(scenario)
        self.config, self.encode = config, observe
        self.refusal_cost, self.cancel_cost = refusal_cost, cancel_cost
        self.terminal = None

    def observe(self):
        return self.encode(tuple(self.state.items()))

    def step(self, action):
        before = len(self.env.trace())
        self.state = self.env.step(action)
        refused = sum(item["activity"] == "receive_refusal" for item in self.env.trace()[before:])
        reward = -self.refusal_cost * refused
        self.profit_reward = 0.
        if self.state is None:
            self.terminal = self.env.result()
            self.profit_reward = self.terminal["profit"] / self.config["profit_scale"]
            reward += self.terminal["profit"] - self.cancel_cost * self.terminal["cancelled"]
        return reward / self.config["profit_scale"], self.state is None

    def result(self):
        values = self.env.result() if self.terminal is None else self.terminal
        return values | {"repeat_refusals": max(0, values["refusals"] - 1)}

    def audit(self, reward):
        row = self.result()
        expected = (row["profit"] - self.refusal_cost * row["refusals"] - self.cancel_cost * row["cancelled"]) / self.config["profit_scale"]
        np.testing.assert_allclose(reward, expected, atol=1e-9)


def run(out, config_path, settings, master, seed, seed_index, mode, threads, torch, progress):
    cfg = json.loads(config_path.read_text()) | {"seed": master}
    rates = cfg["rates"]
    engine = IndependentNegotiationEnvironment(cfg)
    start = settings["scenario_base"] + seed_index * settings["seed_stride"]
    if mode == "smoke":
        start += settings["smoke_offset"]
    ranges = {name: range(start + offset, start + offset + settings[name]) for name, offset in settings["phase_offsets"].items()}
    ordered = sorted(ranges.values(), key=lambda stream: stream.start)
    if any(first.stop > second.start for first, second in zip(ordered, ordered[1:])) or ordered[-1].stop > start + settings["seed_stride"]:
        raise ValueError("SimBank phase ranges overlap")
    save(out / "data.json", {"parameter_file": cfg["parameter_file"], "sha256": digest(Path(cfg["parameter_file"])),
                             "case_ranges": {name: [r.start, r.stop] for name, r in ranges.items()}})

    def collect(name, phase_seed):
        rng = np.random.default_rng(phase_seed)
        return [row for s in ranges[name] for row in run_policy(engine, s, lambda _: int(rng.integers(len(rates))))[1]]

    def encode(rows):
        records = [state_record(row["state"], rates[row["action"]]) for row in rows]
        return records, targets({name: [row[key] for row in rows] for name, key in TARGETS.items()})

    fit_seed, cal_seed, test_seed, random_seed = np.random.SeedSequence(seed).spawn(4)
    progress("forecasts")
    fit = collect("ml_fit", fit_seed)
    records, labels = encode(fit)
    vector = DictVectorizer(sparse=False).fit(records)
    x = vector.transform(records)
    basic = [i for i, key in enumerate(vector.feature_names_) if key not in ("offer_count", "refusal_count", "previous_rate")]
    calibration, ycal = encode(collect("ml_calibration", cal_seed))
    xcal = vector.transform(calibration)
    acceptance = np.array([row["accepted_now"] for row in fit], bool)
    families = tuple(settings.get("forecast_families", FAMILIES))
    if not families or set(families) - set(FAMILIES):
        raise ValueError(f"Unknown forecast families: {families}")
    full, plain, accepted, calibrators = {}, {}, {}, {}
    for family in families:
        for columns, models in ((np.arange(x.shape[1]), full), (basic, plain)):
            models[family] = {key: estimator(family, key, seed, settings["trees"], threads).fit(x[:, columns], y)
                              for key, (y, _) in labels.items()}
        accepted[family] = estimator(family, "profit", seed, settings["trees"], threads).fit(x[acceptance], labels["profit"][0][acceptance])
        p = predict(full[family]["risk"], xcal, True)[:, 1]
        calibrators[family] = LogisticRegression().fit(calibrate_input(p), ycal["risk"][0])
        if family in ("RandomForest", "XGBoost"):
            for model in (*full[family].values(), *plain[family].values(), accepted[family]):
                model.set_params(n_jobs=1)
    joblib.dump((vector, full, plain, accepted, calibrators), out / "predictors.joblib")

    def calibrated(family, z):
        raw = predict(full[family]["risk"], z, True)[:, 1]
        return calibrators[family].predict_proba(calibrate_input(raw))[:, 1]

    def action_matrix(states):
        return vector.transform([state_record(state, rate) for state in states for rate in rates])

    @lru_cache(maxsize=4096)
    def forecast(family, prefixes):
        states = [dict(items) for items in prefixes]
        if family == "Mechanism":
            return (np.array([[1 - native_acceptance(s, rate) for rate in rates] for s in states]),
                    np.array([[estimated_accepted_profit(s, rate) for rate in rates] for s in states]))
        z = action_matrix(states)
        shape = len(states), len(rates)
        return calibrated(family, z).reshape(shape), predict(accepted[family], z).reshape(shape)

    @lru_cache(maxsize=4096)
    def planner(items, family, penalty):
        def predict_states(states):
            return forecast(family, tuple(tuple(state.items()) for state in states))

        return int(plan([dict(items)], rates, cfg["max_offers"], predict_states, penalty,
                        settings["cancellation_penalty"] if penalty else 0).argmax())

    @lru_cache(maxsize=4096)
    def action_features(items):
        return action_matrix([dict(items)])

    rng = np.random.default_rng(random_seed)
    random_by_case = {case: rng.integers(len(rates), size=cfg["max_offers"]) for case in
                      (*ranges["policy_validation"], *ranges["evaluation"])}
    controls = {"Bank": lambda env: bank_rate(env.state, rates),
                "Fixed_revision": lambda env: max(0, len(rates) - 1 - env.state["offer_count"]),
                "Random": lambda env: int(random_by_case[env.env.scenario][env.state["offer_count"]]),
                **{f"Fixed_{rate:g}": (lambda _env, action=i: action) for i, rate in enumerate(rates)}}
    shaped, profit_controls = {}, ["Bank"]
    for family in (*families, "Mechanism"):
        if family in families:
            controls[f"{family}_greedy"] = lambda env, family=family: int(predict(full[family]["profit"], action_features(tuple(env.state.items()))).argmax())
        penalties = settings["mechanism_penalties"] if family == "Mechanism" else (0, settings["refusal_penalty"])
        for penalty in penalties:
            name = f"{family}_DP_{penalty}"
            controls[name] = lambda env, family=family, penalty=penalty: planner(tuple(env.state.items()), family, penalty)
            if penalty == 0:
                profit_controls.append(name)
            if penalty == settings["refusal_penalty"] and family in families:
                shaped[name] = family

    def summarize(rows):
        return {name: dict(zip(FIELDS, values.mean(axis=0).tolist(), strict=True)) for name, values in rows.items()}

    def evaluate(cases, policies, factory, phase=None):
        result = {name: [] for name in policies}
        for index, case in enumerate(cases, 1):
            for name, policy in policies.items():
                env = factory(case)
                while env.state is not None:
                    env.step(policy(env))
                row = env.result()
                if row["accepted"] + row["cancelled"] != 1 or not row["refusals"] <= row["offers"] <= cfg["max_offers"]:
                    raise ValueError("Invalid SimBank case outcome")
                result[name].append([row[field] for field in FIELDS])
            if phase and index % 1000 == 0:
                progress(f"{phase}_{index}_of_{len(cases)}")
        return {name: np.asarray(values) for name, values in result.items()}

    def env_for(case, observe=None):
        return Negotiation(engine, case, cfg, observe, settings["refusal_penalty"], settings["cancellation_penalty"])

    validation_cases = ranges["policy_validation"]
    validation = evaluate(validation_cases, controls, env_for, "policy_validation")
    summaries = summarize(validation)
    bank = summaries["Bank"]
    anchor = max((name for name in profit_controls if summaries[name]["cancelled"] <= bank["cancelled"] + settings["cancellation_tolerance_vs_bank"]),
                 key=lambda name: summaries[name]["profit"])
    reference = summaries[anchor]
    eligible_controls = [name for name in controls if "_DP_" in name or name.endswith("_greedy")]
    strongest = select_low_refusal(summaries, eligible_controls, reference, bank, settings)
    matched = settings.get("fixed_ml_policy")
    if matched is not None and matched not in shaped:
        raise ValueError(f"Fixed ML policy is unavailable: {matched}")
    matched = matched or select_low_refusal(summaries, list(shaped), reference, bank, settings)
    matched = matched or max(shaped, key=lambda name: summaries[name]["profit"])
    family = shaped[matched]

    @lru_cache(maxsize=16384)
    def observe(items):
        z = action_features(items)
        values = [calibrated(family, z), *(predict(full[family][key], z) / scale for key, scale in
                                         (("count", 3), ("effort", 1440), ("profit", cfg["profit_scale"])))]
        return np.column_stack((z, *values)).astype(np.float32), np.ones(len(rates), bool)

    def factory(case):
        return env_for(case, observe)

    def q_key(env):
        state = env.state
        x, _ = env.observe()
        best = int(x[:, -1].argmax())
        return (state["offer_count"], int(np.searchsorted(rates, state["minimum_rate"] - 1e-9)), best,
                int(np.searchsorted((.25, .5, .75), x[best, -4])))

    progress("rl_training")
    rl = fit_policies(factory, ranges["offline"], ranges["online"], seed, settings, out, controls[matched],
                      q_key, list(range(x.shape[1])), threads, torch)
    validation.update(evaluate(validation_cases, rl, factory))
    summaries = summarize(validation)
    cql = select_low_refusal(summaries, [f"CQL_{alpha:g}" for alpha in settings["cql_alphas"]],
                             reference, bank, settings)
    profit_candidates = [name for name in settings.get("profit_first_candidates", []) if name in summaries]
    matched_summary = summaries[matched]
    minimum_profit = settings.get("minimum_validation_profit_gain_fraction", 0.)
    refusal_tolerance = settings.get("refusal_tolerance_vs_ml", 0.)
    feasible_profit = [name for name in profit_candidates
                       if summaries[name]["profit"] >= (1 + minimum_profit) * matched_summary["profit"]
                       and summaries[name]["refusals"] <= matched_summary["refusals"] + refusal_tolerance
                       and summaries[name]["cancelled"] <= matched_summary["cancelled"]
                           + settings["cancellation_tolerance_vs_ml"]
                       and summaries[name]["cancelled"] <= bank["cancelled"]
                           + settings["cancellation_tolerance_vs_bank"]]
    selected_profit = max(feasible_profit, default=None, key=lambda name: summaries[name]["profit"])
    profit_fallback = selected_profit is None and settings.get("profit_first_fallback_to_ml", False)
    if profit_fallback:
        selected_profit = matched
    freeze = {"matched_ml": matched, "strongest_feasible_ml": strongest, "forecast_family": family,
              "forecast_families": list(families),
              "profit_reference": anchor, "selected_cql": cql,
              "selected_profit_controller": selected_profit, "profit_controller_fallback": profit_fallback,
              "profit_candidates": profit_candidates, "validation": summaries}
    save(out / "freeze.json", freeze)
    progress("heldout_evaluation")
    cases = ranges["evaluation"]
    result = evaluate(cases, controls | rl, factory, "heldout_evaluation")
    np.savez_compressed(out / "heldout.npz", **result)
    test, truth = encode(collect("prediction_test", test_seed))
    xt = vector.transform(test)
    prediction = {"history": scores(full, xt, truth), "without_attempt": scores(plain, xt[:, basic], truth)}
    save(out / "results.json", freeze | {"metrics": {"negotiation": summarize(result)}, "fields": FIELDS,
                                         "case_range": [cases.start, cases.stop], "prediction": prediction})
