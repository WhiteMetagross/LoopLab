"""Cross refusal and cancellation penalties on an identical SimBank replay."""

from functools import lru_cache
from fcntl import LOCK_EX, flock
from itertools import product
from pathlib import Path

import joblib
import numpy as np

from scripts.check_negotiation_randomness import IndependentNegotiationEnvironment
from scripts.simbank_constrained_reward import rewards
from scripts.simbank_timed_override import plan
from scripts.simfactory_tradeoff_replay import read
from src import paper_rl
from src.benchmark_models import predict
from src.domain_loop_data import sha256
from src.domain_loop_simulations import save
from src.negotiation_pilot import calibrate_input, estimated_accepted_profit, native_acceptance
from src.negotiation_support import bank_rate
from src.paper_simbank import FIELDS as NATIVE_FIELDS, Negotiation, state_record
from src.runtime import enable_shared_runtime


FIELDS = (*NATIVE_FIELDS, "first_refusal", "last_selected_rate_zero_if_no_decision", "made_offer")
ARMS = dict(zip(("profit", "cancel", "refuse", "both"), product((0, 1), repeat=2), strict=True))
GROUPS = ("both_accepted", "only_candidate_accepted", "only_reference_accepted", "both_cancelled")


def decode_replay(replay, refusal_cost, cancellation_cost, scale):
    # The source reward stores the two penalties and native terminal profit.
    debt = (replay["profit_reward"] - replay["reward"]) * scale
    cancelled = debt >= cancellation_cost - 1e-5
    refused = debt - cancellation_cost * cancelled >= refusal_cost - 1e-5
    np.testing.assert_allclose(debt, refusal_cost * refused + cancellation_cost * cancelled, atol=1e-5)
    if np.any(cancelled & ~replay["done"]) or np.any(replay["profit_reward"][~replay["done"]] != 0.):
        raise ValueError("Saved penalty signals do not match terminal flags")
    return refused, cancelled


def contributions(candidate, reference):
    accepted = NATIVE_FIELDS.index("accepted")
    a, b = candidate[:, accepted].astype(bool), reference[:, accepted].astype(bool)
    groups = np.column_stack((a & b, a & ~b, ~a & b, ~a & ~b))
    delta = (candidate[:, 0] - reference[:, 0])[:, None]
    parts = groups * delta
    np.testing.assert_allclose(parts.sum(axis=1), delta[:, 0])
    return np.column_stack((parts, groups))


def run(folder, index, seed, config, profile, progress, device):
    enable_shared_runtime()
    import torch

    torch.set_num_threads(1)
    paper_rl.DEVICE = device
    settings = config["bank"]
    master = config["master_seed"]
    paper = read(Path(config["paper_config"]))
    original = paper["simbank"]
    cfg = read(Path(paper["simbank_config"])) | {"seed": master}
    source = Path(config["paper_run"]) / "simbank" / f"seed_{index}"
    if read(source / "complete.json")["seed"] != seed:
        raise ValueError("Frozen SimBank seed does not match")
    family = read(source / "freeze.json")["forecast_family"]
    vector, full, _, accepted, calibrators = joblib.load(source / "predictors.joblib")
    models = full[family]
    scaler = joblib.load(source / "policy_models.joblib")["scaler"]
    rates, scale = cfg["rates"], cfg["profit_scale"]
    engine = IndependentNegotiationEnvironment(cfg)
    refuses, cancels = settings["refusal_penalty"], settings["cancellation_penalty"]

    def action_matrix(states):
        return vector.transform([state_record(state, rate) for state in states for rate in rates])

    def risk(z):
        return calibrators[family].predict_proba(calibrate_input(predict(models["risk"], z, True)[:, 1]))[:, 1]

    def normalize(x):
        return np.clip((x - scaler.mean_) / scaler.scale_, -10., 10.).astype(np.float32)

    @lru_cache(maxsize=24576)
    def observe(items):
        z = action_matrix([dict(items)])
        return normalize(np.column_stack((z, risk(z), predict(models["count"], z) / 3,
                                          predict(models["effort"], z) / 1440, predict(models["profit"], z) / scale)).astype(np.float32)), np.ones(len(rates), bool)

    def model_forecast(states):
        z = action_matrix(states)
        shape = (len(states), len(rates))
        return risk(z).reshape(shape), predict(accepted[family], z).reshape(shape)

    def mechanism_forecast(states):
        return (np.array([[1 - native_acceptance(state, rate) for rate in rates] for state in states]),
                np.array([[estimated_accepted_profit(state, rate) for rate in rates] for state in states]))

    @lru_cache(maxsize=24576)
    def planner_actions(items, method):
        predict_states = model_forecast if method == "ML" else mechanism_forecast

        @lru_cache(maxsize=16)
        def cached(prefixes):
            return predict_states([dict(prefix) for prefix in prefixes])

        def forecast(states):
            return cached(tuple(tuple(state.items()) for state in states))

        return {arm: int(plan([dict(items)], rates, cfg["max_offers"], forecast, r * refuses, c * cancels).argmax())
                for arm, (r, c) in ARMS.items()}

    with np.load(source / "offline_replay.npz") as replay_file:
        replay = {key: replay_file[key][:profile.get("bank_replay_limit")] for key in replay_file.files}
    refused, cancelled = decode_replay(replay, original["refusal_penalty"], original["cancellation_penalty"], scale)
    x, nx = normalize(replay["x"]), normalize(replay["next_x"])
    actors = {"Bank": lambda env: bank_rate(env.state, rates)}
    with (folder.parent / "gpu.lock").open("a") as lock:
        if device == "cuda":
            progress("waiting_for_gpu")
            flock(lock, LOCK_EX)
        for arm, (r, c) in ARMS.items():
            progress(f"cql_{arm}")
            reward = rewards(replay["profit_reward"] * scale, refused, cancelled, r * refuses, c * cancels, scale)
            actor, checkpoint = paper_rl.neural((x, replay["mask"], replay["action"], reward, nx, replay["next_mask"],
                                                replay["done"], reward), seed, profile["bank_updates"], settings["alpha"])
            torch.save(checkpoint, folder / f"CQL_{arm}.pt")
            actors[f"CQL_{arm}"] = lambda env, actor=actor: actor(*env.observe())
            for method in ("ML", "Mechanism"):
                actors[f"{method}_{arm}"] = lambda env, method=method, arm=arm: planner_actions(tuple(env.state.items()), method)[arm]
        if device == "cuda":
            torch.cuda.empty_cache()

    start = settings["scenario_base"] + (0 if "bank_replay_limit" not in profile else settings["seed_stride"] * config["replicates"])
    start += index * settings["seed_stride"]
    stop = start + profile["bank_cases"]
    if stop > start + settings["seed_stride"]:
        raise ValueError("Evaluation crosses a seed range")
    progress("paired_evaluation")
    rows = {name: np.empty((stop - start, len(FIELDS))) for name in actors}
    for position, case in enumerate(range(start, stop)):
        for name, actor in actors.items():
            env = Negotiation(engine, case, cfg, observe, 0, 0)
            last_rate = 0.
            while env.state is not None:
                action = actor(env)
                last_rate = rates[action]
                env.step(action)
            row = env.result()
            if row["accepted"] + row["cancelled"] != 1 or not row["refusals"] <= row["offers"] <= cfg["max_offers"]:
                raise ValueError("Invalid native outcome")
            rows[name][position] = [*(row[field] for field in NATIVE_FIELDS), int(row["refusals"] > 0), last_rate, int(row["offers"] > 0)]
        if (position + 1) % 500 == 0:
            progress(f"paired_evaluation_{position + 1}_of_{stop - start}")

    contrasts = {}
    for method in ("CQL", "ML", "Mechanism"):
        for label, terms in {
            "refusal_main": {"both": .5, "refuse": .5, "cancel": -.5, "profit": -.5},
            "cancellation_main": {"both": .5, "cancel": .5, "refuse": -.5, "profit": -.5},
            "interaction": {"both": 1, "profit": 1, "refuse": -1, "cancel": -1},
            "both_minus_profit": {"both": 1, "profit": -1},
        }.items():
            contrasts[f"{method}_{label}"] = {f"{method}_{arm}": weight for arm, weight in terms.items()}
    contrasts |= {f"CQL_both_minus_{method}_both": {"CQL_both": 1, f"{method}_both": -1} for method in ("ML", "Mechanism")}
    decomp = {name: contributions(rows["CQL_both"], rows[name]) for name in ("CQL_profit", "Mechanism_both", "ML_both")}
    save(folder / "design.json", {"fields": FIELDS, "contrasts": contrasts, "replay_rows": len(x), "seed": seed,
                                  "source": str(source), "forecast_family": family, "case_range": [start, stop],
                                  "data": {str(path): sha256(path.read_bytes()) for path in (
                                      source / "offline_replay.npz", source / "freeze.json", source / "predictors.joblib",
                                      source / "policy_models.joblib", Path(cfg["parameter_file"]))},
                                  "profit_decomposition": {name: dict(zip((*GROUPS, *(f"share_{key}" for key in GROUPS)),
                                                                         values.mean(axis=0).tolist(), strict=True))
                                                           for name, values in decomp.items()}})
    np.savez_compressed(folder / "profit_decomposition.npz", **decomp)
    np.savez_compressed(folder / "paired.npz", **rows)
