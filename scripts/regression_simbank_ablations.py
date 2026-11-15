"""Regression focused SimBank reward, selector, and logging ablations."""

from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

from scripts.check_negotiation_randomness import IndependentNegotiationEnvironment
from scripts.simbank_constrained_reward import rewards
from scripts.simbank_logging_followup import run as run_logging
from scripts.simbank_loop_ablation import ARMS, decode_replay
from scripts.simbank_timed_override import plan
from src import paper_rl
from src.benchmark_models import predict
from src.domain_loop_data import sha256
from src.domain_loop_simulations import save
from src.negotiation_pilot import calibrate_input, estimated_accepted_profit, native_acceptance
from src.negotiation_support import bank_rate
from src.paper_simbank import FIELDS, Negotiation, state_record


def load_neural(path, width):
    import torch
    net = paper_rl.network(width)
    net.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    net.eval()
    return paper_rl.greedy(net)


def run(folder, index, seed, config, profile, progress, device):
    import json
    import torch

    torch.set_num_threads(1)
    paper_rl.DEVICE = device
    paper_rl.THREADS = 1
    source = Path(config["sources"]["simbank"]) / f"seed_{index}"
    freeze = json.loads((source / "freeze.json").read_text())
    if freeze["forecast_family"] != "Regression":
        raise ValueError("SimBank ablation source must use Regression forecasts")
    paper = json.loads(Path(config["focused_config"]).read_text())
    settings = paper["simbank"]
    cfg = json.loads(Path(paper["simbank_config"]).read_text()) | {"seed": config["master_seed"]}
    rates, scale = cfg["rates"], cfg["profit_scale"]
    engine = IndependentNegotiationEnvironment(cfg)
    vector, full, _, accepted, calibrators = joblib.load(source / "predictors.joblib")
    models = full["Regression"]
    saved = joblib.load(source / "policy_models.joblib")
    scaler = saved["scaler"]

    def action_matrix(states):
        return vector.transform([state_record(state, rate) for state in states for rate in rates])

    def risk(z):
        raw = predict(models["risk"], z, True)[:, 1]
        return calibrators["Regression"].predict_proba(calibrate_input(raw))[:, 1]

    def normalize(x):
        return np.clip((x - scaler.mean_) / scaler.scale_, -10, 10).astype(np.float32)

    @lru_cache(maxsize=24576)
    def observe(items):
        z = action_matrix([dict(items)])
        values = np.column_stack((z, risk(z), predict(models["count"], z) / 3,
                                  predict(models["effort"], z) / 1440,
                                  predict(models["profit"], z) / scale)).astype(np.float32)
        return normalize(values), np.ones(len(rates), bool)

    def model_forecast(states):
        z = action_matrix(states)
        shape = len(states), len(rates)
        return risk(z).reshape(shape), predict(accepted["Regression"], z).reshape(shape)

    @lru_cache(maxsize=24576)
    def regression_action(items):
        states = [dict(items)]
        return int(plan(states, rates, cfg["max_offers"], model_forecast,
                        settings["refusal_penalty"], settings["cancellation_penalty"]).argmax())

    def actor_from_model(model):
        def actor(x, mask):
            q = model.predict(x, thread_count=1)
            return int(np.where(mask, q, -np.inf).argmax())
        return actor

    with np.load(source / "offline_replay.npz") as replay_file:
        raw = {key: replay_file[key][:profile.get("bank_replay_limit")] for key in replay_file.files}
    refused, cancelled = decode_replay(raw, settings["refusal_penalty"], settings["cancellation_penalty"], scale)
    x, nx = normalize(raw["x"]), normalize(raw["next_x"])
    base_replay = (x, raw["mask"], raw["action"], raw["reward"], nx, raw["next_mask"],
                   raw["done"], raw["return"])
    actors = {"Bank": lambda env: bank_rate(env.state, rates),
              "Regression_DP_500": lambda env: regression_action(tuple(env.state.items()))}
    reward_design = {}
    for label, (refuse_flag, cancel_flag) in ARMS.items():
        progress(f"reward_{label}")
        reward = rewards(raw["profit_reward"] * scale, refused, cancelled,
                         refuse_flag * settings["refusal_penalty"],
                         cancel_flag * settings["cancellation_penalty"], scale)
        replay = (x, raw["mask"], raw["action"], reward, nx, raw["next_mask"], raw["done"], reward)
        actor, checkpoint = paper_rl.neural(replay, seed, profile["bank_updates"], config["bank"]["alpha"])
        torch.save(checkpoint, folder / f"CQL_{label}.pt")
        actors[f"CQL_{label}"] = lambda env, actor=actor: actor(*env.observe())
        reward_design[label] = {"refusal_penalty": refuse_flag * settings["refusal_penalty"],
                                "cancellation_penalty": cancel_flag * settings["cancellation_penalty"]}

    profit_replay = (x, raw["mask"], raw["action"], raw["profit_reward"], nx,
                     raw["next_mask"], raw["done"], raw["profit_reward"])
    fitted_actor, fitted_model = paper_rl.fitted(profit_replay, seed, profile["fitted_q_rounds"], profile["trees"])
    joblib.dump(fitted_model, folder / "profit_fitted_q.joblib")
    actors["Profit_FittedQ"] = lambda env: fitted_actor(*env.observe())
    ensemble = saved["FittedQ_profit_ensemble"]

    def ensemble_actor(x_values, mask):
        q = np.mean([model.predict(x_values, thread_count=1) for model in ensemble], axis=0)
        return int(np.where(mask, q, -np.inf).argmax())

    actors["Profit_FittedQ_ensemble"] = lambda env: ensemble_actor(*env.observe())
    history_columns = np.arange(len(vector.feature_names_))
    history_scaler = saved["history_scaler"]

    def history_observe(items):
        z = action_matrix([dict(items)])[:, history_columns]
        return np.clip((z - history_scaler.mean_) / history_scaler.scale_, -10, 10).astype(np.float32), np.ones(len(rates), bool)

    history_actor = load_neural(source / "CQL_history.pt", len(history_columns))
    actors["CQL_history"] = lambda env: history_actor(*history_observe(tuple(env.state.items())))

    def evaluate(cases, policies, phase):
        result = {name: np.empty((len(cases), len(FIELDS))) for name in policies}
        for position, case in enumerate(cases):
            for name, actor in policies.items():
                env = Negotiation(engine, case, cfg, observe, 0, 0)
                while env.state is not None:
                    env.step(actor(env))
                row = env.result()
                result[name][position] = [row[field] for field in FIELDS]
            if (position + 1) % 1000 == 0:
                progress(f"{phase}_{position + 1}")
        return result

    base = config["bank"]["scenario_base"] + index * config["bank"]["seed_stride"]
    if profile.get("smoke_offset"):
        base += profile["smoke_offset"]
    validation_cases = range(base, base + profile["bank_validation_cases"])
    test_start = base + config["bank"]["test_offset"]
    test_cases = range(test_start, test_start + profile["bank_cases"])
    if validation_cases.stop > test_cases.start or test_cases.stop > base + config["bank"]["seed_stride"]:
        raise ValueError("SimBank ablation ranges overlap")
    progress("selector_validation")
    validation = evaluate(validation_cases, actors, "selector_validation")
    means = {name: values.mean(axis=0) for name, values in validation.items()}
    profit_index, refusal_index, cancel_index = (FIELDS.index(name) for name in ("profit", "refusals", "cancelled"))
    candidates = ["CQL_both", "Profit_FittedQ", "Profit_FittedQ_ensemble"]
    unconstrained = max(candidates, key=lambda name: means[name][profit_index])
    ml = means["Regression_DP_500"]
    bank = means["Bank"]
    admissible = [name for name in candidates
                  if means[name][profit_index] >= (1 + settings["minimum_validation_profit_gain_fraction"]) * ml[profit_index]
                  and means[name][refusal_index] <= ml[refusal_index] + settings["refusal_tolerance_vs_ml"]
                  and means[name][cancel_index] <= ml[cancel_index] + settings["cancellation_tolerance_vs_ml"]
                  and means[name][cancel_index] <= bank[cancel_index] + settings["cancellation_tolerance_vs_bank"]]
    constrained = max(admissible, default="Regression_DP_500", key=lambda name: means[name][profit_index])
    actors["Unconstrained_selector"] = actors[unconstrained]
    actors["Constrained_selector"] = actors[constrained]
    progress("paired_evaluation")
    rows = evaluate(test_cases, actors, "paired_evaluation")

    logging_folder = folder / "logging"
    logging_folder.mkdir()
    logging_profile = profile | {"namespace": profile["logging_namespace"]}
    run_logging(logging_folder, index, seed, config, logging_profile, progress, device)
    with np.load(logging_folder / "paired.npz") as logging:
        rows.update({f"logging_{name}": logging[name] for name in logging.files})

    contrasts = {
        "both_minus_profit": {"CQL_both": 1.0, "CQL_profit": -1.0},
        "refusal_main": {"CQL_both": .5, "CQL_refuse": .5, "CQL_cancel": -.5, "CQL_profit": -.5},
        "cancellation_main": {"CQL_both": .5, "CQL_cancel": .5, "CQL_refuse": -.5, "CQL_profit": -.5},
        "reward_interaction": {"CQL_both": 1.0, "CQL_profit": 1.0, "CQL_refuse": -1.0, "CQL_cancel": -1.0},
        "ensemble_minus_single": {"Profit_FittedQ_ensemble": 1.0, "Profit_FittedQ": -1.0},
        "constrained_minus_unconstrained": {"Constrained_selector": 1.0, "Unconstrained_selector": -1.0},
        "constrained_minus_fixed_CQL": {"Constrained_selector": 1.0, "CQL_both": -1.0},
        "full_CQL_minus_history": {"CQL_both": 1.0, "CQL_history": -1.0},
        "logging_CQL_biased_minus_random": {"logging_CQL_biased": 1.0, "logging_CQL_random": -1.0},
    }
    save(folder / "design.json", {"fields": FIELDS, "contrasts": contrasts, "seed": seed,
         "forecast_family": "Regression", "source": str(source), "reward_design": reward_design,
         "selector": {"unconstrained": unconstrained, "constrained": constrained,
                      "fallback": constrained == "Regression_DP_500", "admissible": admissible},
         "validation_range": [validation_cases.start, validation_cases.stop],
         "case_range": [test_cases.start, test_cases.stop],
         "source_sha256": {str(path): sha256(path.read_bytes()) for path in
                           (source / "freeze.json", source / "predictors.joblib",
                            source / "offline_replay.npz", source / "policy_models.joblib")}})
    np.savez_compressed(folder / "paired.npz", **rows)
