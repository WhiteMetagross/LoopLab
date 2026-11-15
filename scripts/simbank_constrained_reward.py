"""Test CQL refusal reduction against matched constrained ML planning."""
import argparse
from functools import lru_cache
import json
from pathlib import Path
import signal
import time

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from scripts.check_negotiation_randomness import IndependentNegotiationEnvironment
from src import paper_rl
from src.loop_feasibility import digest, write_json
from src.negotiation_pilot import (action_rows, bank_rate, calibrate_input,
                                   estimated_accepted_profit, native_acceptance, run_policy)
from src.negotiation_support import cancel_profit, refusal_successor
from src.benchmark_models import predict, seeds
from src.paper_utils import summarize
from src.runtime import enable_shared_runtime


def rewards(profit, refusal, cancelled, refusal_cost, cancellation_cost, scale):
    return (np.asarray(profit) - refusal_cost * np.asarray(refusal) -
            cancellation_cost * np.asarray(cancelled)) / scale


def select_low_refusal(summary, names, ref, bank, settings):
    feasible = [name for name in names if summary[name]["profit"] >=
                ref["profit"] * (1 - settings["profit_tolerance_fraction"])
                and summary[name]["cancelled"] <= min(
                    ref["cancelled"] + settings["cancellation_tolerance_vs_ml"],
                    bank["cancelled"] + settings["cancellation_tolerance_vs_bank"])]
    return min(feasible, key=lambda name: (summary[name]["refusals"], -summary[name]["profit"])) if feasible else None


def paired(a, b, metric):
    diff = np.array([x[metric] - y[metric] for x, y in zip(a, b, strict=True)])
    mean = diff.mean()
    se = diff.std(ddof=1) / np.sqrt(len(diff))
    return {"difference": float(mean), "lower_95_one_sided": float(mean - 1.644854 * se),
            "upper_95_one_sided": float(mean + 1.644854 * se),
            "two_sided_95": [float(mean - 1.96 * se), float(mean + 1.96 * se)]}


def smoke():
    np.testing.assert_allclose(rewards([0, 10000], [1, 0], [0, 1], 500, 2000, 10000), [-.05, .8])
    cfg = {"profit_tolerance_fraction": .005, "cancellation_tolerance_vs_ml": .005,
           "cancellation_tolerance_vs_bank": .02}
    ref = {"profit": 100., "cancelled": .2, "refusals": .3}
    table = {"low_profit": {**ref, "profit": 99., "refusals": 0.},
             "high_cancel": {**ref, "cancelled": .21, "refusals": 0.},
             "valid": {**ref, "profit": 99.6, "refusals": .2}}
    assert select_low_refusal(table, list(table), ref, ref, cfg) == "valid"
    print("Transition reward and selection constraint checks passed.")


def run(out, config_path, device):
    started = time.perf_counter()
    settings = json.loads(config_path.read_text())
    signal.alarm(settings["stage_cap_seconds"])
    out.mkdir(parents=True)
    root = Path("runs/paper_full")
    master = json.loads((root / "manifest.json").read_text())["config"]["master_seed"]
    seed = seeds(master, 1)[0]
    parent = root / "simbank" / str(seed)
    cfg = json.loads(Path("configs/negotiation_independent.json").read_text())
    cfg["seed"] = master
    rates, scale = cfg["rates"], cfg["profit_scale"]
    engine = IndependentNegotiationEnvironment(cfg)
    vector, fitted, _, accept, calibrator = joblib.load(parent / "predictors.joblib")
    ml = fitted["CatBoost"]
    penalties = settings["refusal_penalties"]
    ablation = settings.get("fixed_ablation_penalty")
    if ablation is not None and ablation not in penalties:
        raise ValueError("The fixed ablation penalty must be in the refusal penalty grid.")
    cancel_cost = settings["cancellation_penalty"]
    ranges = {k: range(settings[f"{k}_start"], settings[f"{k}_start"] + settings[f"{k}_cases"])
              for k in ("fit", "validation", "evaluation")}
    assert all(r.stop <= s.start or s.stop <= r.start for i, r in enumerate(ranges.values())
               for s in list(ranges.values())[i + 1:])
    enable_shared_runtime()
    import torch
    torch.set_num_threads(1)
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable.")
    paper_rl.DEVICE = device

    def encode(states):
        return vector.transform(action_rows(states, rates).to_dict("records"))

    def calibrated(X):
        return calibrator.predict_proba(calibrate_input(predict(ml["risk"], X, True)[:, 1]))[:, 1]

    @lru_cache(maxsize=22000)
    def state_features(items):
        X = encode([dict(items)])
        return np.column_stack([X, calibrated(X), predict(ml["count"], X) / 3,
                                predict(ml["effort"], X) / 1440,
                                predict(ml["profit"], X) / 10000]).astype(np.float32)

    def plan(states, penalty, native):
        n, k = len(states), len(rates)
        if native:
            risk = np.array([[1 - native_acceptance(s, a) for a in rates] for s in states])
            accepted = np.array([[estimated_accepted_profit(s, a) for a in rates] for s in states])
        else:
            X = encode(states)
            risk, accepted = calibrated(X).reshape(n, k), predict(accept, X).reshape(n, k)
        cancel = cancel_cost if penalty else 0.
        q = (1 - risk) * accepted - penalty * risk
        successors, locations = [], []
        for i, state in enumerate(states):
            for a, rate in enumerate(rates):
                if rate < state["minimum_rate"] - 1e-9:
                    q[i, a] = cancel_profit(state) - cancel
                elif state["offer_count"] + 1 == cfg["max_offers"]:
                    q[i, a] += risk[i, a] * (cancel_profit(state, refusal=True) - cancel)
                else:
                    successors.append(refusal_successor(state, rate))
                    locations.append((i, a))
        if successors:
            future = plan(successors, penalty, native).max(axis=1)
            for (i, a), value in zip(locations, future, strict=True):
                q[i, a] += risk[i, a] * value
        return q

    @lru_cache(maxsize=36000)
    def values(items, penalty, native):
        return plan([dict(items)], penalty, native)[0]

    def planner(penalty, native):
        return lambda state: int(values(tuple(state.items()), penalty, native).argmax())

    bank = lambda state: bank_rate(state, rates)
    fixed = {f"{family}_{penalty}": planner(penalty, native)
             for family, native in (("ML", False), ("Mechanism", True)) for penalty in [0, *penalties]}
    rng = np.random.default_rng(seed)
    X, next_X, actions, terminal, native_profit, refusal, cancelled, stages = [], [], [], [], [], [], [], []
    for scenario in ranges["fit"]:
        style = rng.choice(3, p=[.55, .15, .3])
        behavior = ((lambda _: int(rng.integers(len(rates)))) if style == 0 else
                    bank if style == 1 else fixed[f"ML_{penalties[scenario % len(penalties)]}"])
        result, transitions = run_policy(engine, scenario, behavior)
        assert sum(row["immediate_refusal"] for row in transitions) == result["refusals"]
        for row in transitions:
            state = row["state"]
            x = state_features(tuple(state.items()))
            nxt = row["next_state"]
            done = nxt is None
            X.append(x)
            next_X.append(np.zeros_like(x) if done else state_features(tuple(nxt.items())))
            actions.append(row["action"])
            terminal.append(done)
            native_profit.append(result["profit"] if done else 0.)
            refusal.append(row["immediate_refusal"])
            cancelled.append(result["cancelled"] if done else 0)
            stages.append(state["offer_count"])
    X, next_X = np.asarray(X), np.asarray(next_X)
    actions, terminal = np.asarray(actions), np.asarray(terminal)
    mask = np.ones(X.shape[:2], dtype=bool)
    next_mask = mask & ~terminal[:, None]
    scaler = StandardScaler().fit(X.reshape(-1, X.shape[-1]))

    def normalize(x):
        return np.clip((x - scaler.mean_) / scaler.scale_, -10, 10).astype(np.float32)

    x, nx = normalize(X), normalize(next_X)
    joblib.dump(scaler, out / "scaler.joblib")
    np.savez_compressed(out / "replay.npz", x=X, next_x=next_X, action=actions, terminal=terminal,
                        profit=native_profit, refusals=refusal, cancelled=cancelled, stage=stages)
    actors = {}
    for penalty in (penalties if ablation is None else [0, *penalties]):
        r = rewards(native_profit, refusal, cancelled, penalty, cancel_cost if penalty else 0., scale)
        actor, checkpoint = paper_rl.neural((x, mask, actions, r, nx, next_mask, terminal, r),
                                             seed, settings["cql_updates"], settings["cql_alpha"])
        actors[penalty] = actor
        torch.save(checkpoint, out / f"cql_{penalty}.pt")
    print(f"Trained {len(actors)} reward variants on {len(actions)} shared transitions.", flush=True)

    def cql(penalty, guard):
        def act(state):
            items = tuple(state.items())
            observation = normalize(state_features(items))
            permitted = np.ones(len(rates), bool)
            if guard:
                profit = values(items, 0, False)
                permitted = profit >= profit.max() - settings["guard_max_planned_profit_gap"]
            return actors[penalty](observation, permitted)
        return act

    cql_grid = {f"CQL_{p}_{guard}": cql(p, guard) for p in penalties for guard in (False, True)}
    audit_controls = {} if ablation is None else {"Profit_CQL_same_replay": cql(0, False),
                                                  "Fixed_loop_CQL": cql(ablation, False)}

    def evaluate(scenarios, policies):
        return {name: [run_policy(engine, s, actor)[0] for s in scenarios] for name, actor in policies.items()}

    validation = evaluate(ranges["validation"], {"Bank": bank, **fixed, **cql_grid,
                                                  **{k: v for k, v in audit_controls.items() if k == "Profit_CQL_same_replay"}})
    metrics = {k: summarize(v) for k, v in validation.items()}
    bank_limit = metrics["Bank"]["cancelled"] + settings["cancellation_tolerance_vs_bank"]
    anchors = ["Bank"] + [k for k in ("ML_0", "Mechanism_0") if metrics[k]["cancelled"] <= bank_limit]
    anchor = max(anchors, key=lambda k: metrics[k]["profit"])
    ref = metrics[anchor]
    ml_choice = select_low_refusal(metrics, [*fixed, anchor], ref, metrics["Bank"], settings)
    rl_choice = select_low_refusal(metrics, list(cql_grid), ref, metrics["Bank"], settings)
    write_json(out / "freeze.json", {"profit_reference": anchor, "ml_choice": ml_choice,
                                    "cql_choice": rl_choice, "validation": metrics})
    keep, scale_old = joblib.load(parent / "policies.joblib")["transform_forecast"]
    net = paper_rl.network(len(keep))
    net.load_state_dict(torch.load(parent / "CQL_forecast.pt", map_location="cpu", weights_only=True))
    frozen = paper_rl.greedy(net)

    def frozen_cql(state):
        features = state_features(tuple(state.items()))[:, keep]
        return frozen(np.clip((features - scale_old.mean_) / scale_old.scale_, -10, 10), np.ones(len(rates), bool))

    controls = {"Bank": bank, **fixed}
    policies = {"Bank": bank, "Profit_reference": controls[anchor],
                "Loop_ML": controls[ml_choice], "Frozen_CQL": frozen_cql,
                "Loop_CQL": cql_grid[rl_choice] if rl_choice else controls[anchor], **audit_controls}
    rows = evaluate(ranges["evaluation"], policies)
    pairs = [
            ("Loop_CQL", "Loop_ML"), ("Loop_CQL", "Profit_reference"), ("Loop_CQL", "Bank"),
            ("Loop_ML", "Profit_reference"), ("Frozen_CQL", "Loop_ML")]
    if ablation is not None:
        pairs.extend((name, base) for name, base in (("Fixed_loop_CQL", "Profit_CQL_same_replay"),
                                                      ("Fixed_loop_CQL", "Loop_ML"),
                                                      ("Profit_CQL_same_replay", "Loop_ML")))
    comparison = {f"{a}_vs_{b}": {k: paired(rows[a], rows[b], k) for k in (
        "profit", "refusals", "negotiation_effort", "cancelled")} for a, b in pairs}
    final = {k: summarize(v) for k, v in rows.items()}
    delta = comparison["Loop_CQL_vs_Loop_ML"]
    guard = comparison["Loop_CQL_vs_Profit_reference"]
    tolerance = settings["profit_tolerance_fraction"]
    checks = {"refusal_reduction": delta["refusals"]["upper_95_one_sided"] < 0,
              "profit_preserved_vs_loop_ml": delta["profit"]["lower_95_one_sided"] >= -tolerance * final["Loop_ML"]["profit"],
              "profit_preserved_vs_profit_reference": guard["profit"]["lower_95_one_sided"] >= -tolerance * final["Profit_reference"]["profit"],
              "cancellation_vs_profit_reference": guard["cancelled"]["upper_95_one_sided"] <= settings["cancellation_tolerance_vs_ml"],
              "cancellation_vs_bank": comparison["Loop_CQL_vs_Bank"]["cancelled"]["upper_95_one_sided"] <= settings["cancellation_tolerance_vs_bank"]}
    write_json(out / "evaluation_rows.json", rows)
    report = {"seed": seed, "fit_rows": len(actions), "stage_counts": np.bincount(stages).tolist(),
              "action_stage_counts": np.zeros((cfg["max_offers"], len(rates)), dtype=int).tolist(),
              "selection": {"profit_reference": anchor, "ml_choice": ml_choice, "cql_choice": rl_choice},
              "results": final, "paired": comparison, "checks": checks,
              "passed": bool(rl_choice and all(checks.values())), "seconds": time.perf_counter() - started,
              "settings": settings, "device": device, "torch": torch.__version__,
              "sha256": {str(p): digest(p) for p in (Path(__file__), config_path, parent / "predictors.joblib")}}
    support = np.zeros((cfg["max_offers"], len(rates)), dtype=int)
    np.add.at(support, (stages, actions), 1)
    report["action_stage_counts"] = support.tolist()
    write_json(out / "results.json", report)
    print(json.dumps({k: report[k] for k in ("selection", "fit_rows", "stage_counts", "results",
                                             "checks", "passed", "seconds")}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--config", type=Path, default=Path("configs/simbank_constrained_reward.json"))
    parser.add_argument("--out", type=Path, default=Path("runs/simbank_constrained_reward_20261104"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        smoke() if args.smoke else run(args.out, args.config, args.device)
