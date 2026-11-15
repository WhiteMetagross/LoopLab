"""Compare fixed timing gates for a frozen SimBank policy on fresh paired cases."""

import argparse
from collections import Counter
from functools import lru_cache
import json
from pathlib import Path
import platform
import signal
from time import perf_counter

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from scripts.check_negotiation_randomness import IndependentNegotiationEnvironment
from scripts.simbank_constrained_reward import paired, select_low_refusal
from src import paper_rl
from src.benchmark_models import predict, seeds
from src.loop_feasibility import digest, write_json
from src.negotiation_pilot import action_rows, calibrate_input, estimated_accepted_profit, native_acceptance
from src.negotiation_support import bank_rate, cancel_profit, refusal_successor, run_policy
from src.paper_utils import summarize
from src.runtime import enable_shared_runtime


def plan(states, rates, horizon, forecast, refusal_cost, cancellation_cost):
    risk, accepted = forecast(states)
    values = (1. - risk) * accepted - refusal_cost * risk
    later, locations = [], []
    for i, state in enumerate(states):
        for a, rate in enumerate(rates):
            if rate < state["minimum_rate"] - 1e-9:
                values[i, a] = cancel_profit(state) - cancellation_cost
            elif state["offer_count"] + 1 == horizon:
                values[i, a] += risk[i, a] * (cancel_profit(state, refusal=True) - cancellation_cost)
            else:
                later.append(refusal_successor(state, rate))
                locations.append((i, a))
    if later:
        continuations = plan(later, rates, horizon, forecast, refusal_cost, cancellation_cost).max(axis=1)
        for (i, a), continuation in zip(locations, continuations, strict=True):
            values[i, a] += risk[i, a] * continuation
    return values


def override(state, first_offer, base, profit, rates, gap, actor):
    if first_offer != (state["offer_count"] == 0):
        return base
    allowed = (rates >= state["minimum_rate"] - 1e-9) & (profit >= profit[base] - gap)
    return actor(allowed) if allowed[base] and allowed.sum() > 1 else base


def gate(results, candidate, comparator, anchor, settings):
    comparisons = {name: {metric: paired(results[candidate], results[name], metric)
                          for metric in ("refusals", "repeat_refusals", "profit", "cancelled", "negotiation_effort")}
                   for name in dict.fromkeys((comparator, anchor, "Bank"))}
    delta = comparisons[comparator]["refusals"]
    controls = {comparator, anchor}
    checks = {
        "refusal_reduction": delta["difference"] <= -settings["minimum_refusal_reduction"] and delta["two_sided_95"][1] < 0,
        "profit_preserved": all(comparisons[name]["profit"]["lower_95_one_sided"] >=
                                -settings["profit_tolerance_fraction"] * np.mean([r["profit"] for r in results[name]])
                                for name in controls),
        "cancellation_preserved": all(comparisons[name]["cancelled"]["upper_95_one_sided"] <=
                                      settings["cancellation_tolerance_vs_ml"] for name in controls),
        "cancellation_vs_bank": comparisons["Bank"]["cancelled"]["upper_95_one_sided"] <= settings["cancellation_tolerance_vs_bank"],
    }
    return {"checks": checks, "passed": all(checks.values()), "paired": comparisons}


def run(out, config_path):
    started = perf_counter()
    settings = json.loads(config_path.read_text())
    signal.alarm(settings["stage_cap_seconds"])
    out.mkdir(parents=True)
    root = Path("runs/paper_full")
    source = json.loads((root / "manifest.json").read_text())["config"]
    master = source["master_seed"]
    seed = seeds(master, 1)[0]
    parent = root / "simbank" / str(seed)
    sim_config = Path(source["sim_config"])
    cfg = json.loads(sim_config.read_text()) | {"seed": master}
    ranges = {phase: range(settings[f"{phase}_start"], settings[f"{phase}_start"] + settings[f"{phase}_cases"])
              for phase in ("validation", "evaluation")}
    assert ranges["validation"].start >= 44_000_000 and ranges["validation"].stop <= ranges["evaluation"].start
    rates = np.array(cfg["rates"])
    engine = IndependentNegotiationEnvironment(cfg)
    vector, models, _, accepted, calibrator = joblib.load(parent / "predictors.joblib")
    keep, scaler = joblib.load(parent / "policies.joblib")["transform_forecast"]
    heads = models["CatBoost"]
    enable_shared_runtime()
    import torch
    torch.set_num_threads(1)
    actor = paper_rl.network(len(keep))
    actor.load_state_dict(torch.load(parent / "CQL_forecast.pt", map_location="cpu", weights_only=True))
    cql = paper_rl.greedy(actor)

    def encode(states):
        return vector.transform(action_rows(states, rates).to_dict("records"))

    def calibrated(x):
        return calibrator.predict_proba(calibrate_input(predict(heads["risk"], x, True)[:, 1]))[:, 1]

    def learned(states):
        x = encode(states)
        shape = (len(states), len(rates))
        return calibrated(x).reshape(shape), predict(accepted, x).reshape(shape)

    def mechanism(states):
        return (np.array([[1 - native_acceptance(state, rate) for rate in rates] for state in states]),
                np.array([[estimated_accepted_profit(state, rate) for rate in rates] for state in states]))

    @lru_cache(maxsize=16_000)
    def values(items, family, penalty):
        forecast = learned if family == "ML" else mechanism
        cancel = settings["cancellation_penalty"] if penalty else 0.
        return plan([dict(items)], rates, cfg["max_offers"], forecast, penalty, cancel)[0]

    def planner(family, penalty):
        return lambda state: int(values(tuple(state.items()), family, penalty).argmax())

    @lru_cache(maxsize=16_000)
    def features(items):
        x = encode([dict(items)])
        forecasts = [calibrated(x), *(predict(heads[k], x) / scale for k, scale in
                                      (("count", 3), ("effort", 1440), ("profit", 10000)))]
        z = np.column_stack([x, *forecasts]).astype(np.float32)[:, keep]
        return np.clip((z - scaler.mean_) / scaler.scale_, -10, 10)

    def all_offers(state):
        return cql(features(tuple(state.items())), np.ones(len(rates), bool))

    def evaluate(scenarios, policies):
        all_rows, diagnostics = {}, {}
        for name, policy in policies.items():
            rows, decisions = [], Counter()
            for scenario in scenarios:
                result, trace = run_policy(engine, scenario, policy)
                decisions.update(row["state"]["offer_count"] for row in trace)
                result.update(case=scenario, actions=[row["action"] for row in trace],
                              repeat_refusals=max(0, result["refusals"] - 1))
                rows.append(result)
            all_rows[name], diagnostics[name] = rows, dict(decisions)
        return all_rows, diagnostics

    def means(rows):
        return {key: summarize([{field: value for field, value in row.items() if field not in {"case", "actions"}}
                                for row in block]) for key, block in rows.items()}

    bank = lambda state: bank_rate(state, rates)
    controls = {f"{family}_{penalty}": planner(family, penalty)
                for family in ("ML", "Mechanism") for penalty in settings["refusal_penalties"]}
    val, _ = evaluate(ranges["validation"], {"Bank": bank, **controls})
    stats = means(val)
    limit = stats["Bank"]["cancelled"] + settings["cancellation_tolerance_vs_bank"]
    anchor = max((key for key in ("Bank", "ML_0", "Mechanism_0") if stats[key]["cancelled"] <= limit),
                 key=lambda key: stats[key]["profit"])
    learned_anchor = select_low_refusal(stats, [key for key in controls if key.startswith("ML_")],
                                       stats["ML_0"], stats["Bank"], settings) or "ML_0"
    strongest = select_low_refusal(stats, [*controls, "Bank"], stats[anchor], stats["Bank"], settings) or anchor
    base = controls[learned_anchor]

    def timed(first_offer):
        def policy(state):
            action = base(state)
            if first_offer != (state["offer_count"] == 0):
                return action
            items = tuple(state.items())
            return override(state, first_offer, action, values(items, "ML", 0), rates,
                            settings["planned_profit_gap"], lambda mask: cql(features(items), mask))
        return policy

    variants = {"CQL_first_offer": timed(True), "CQL_after_refusal": timed(False), "CQL_all_offers": all_offers}
    val_hybrids, _ = evaluate(ranges["validation"], variants)
    stats.update(means(val_hybrids))
    selected = select_low_refusal(stats, list(variants)[:2], stats[anchor], stats["Bank"], settings)
    hashes = (config_path, sim_config, Path(cfg["parameter_file"]), Path(__file__),
              Path("scripts/check_negotiation_randomness.py"), Path("src/negotiation_support.py"),
              Path("src/negotiation_pilot.py"), Path("src/paper_rl.py"), Path("src/benchmark_models.py"),
              *(parent / name for name in ("predictors.joblib", "policies.joblib", "CQL_forecast.pt")))
    freeze = {"seed": seed, "master": master, "settings": settings, "strongest": strongest,
              "matched": learned_anchor, "profit_reference": anchor, "selected_timed_policy": selected,
              "validation": stats, "source_sha256": {str(p): digest(p) for p in hashes}}
    write_json(out / "freeze.json", freeze)
    print(f"Frozen validation choices: matched={learned_anchor}, strongest={strongest}, timed={selected}", flush=True)

    policies = {"Bank": bank, **{key: controls[key] for key in dict.fromkeys((learned_anchor, strongest, anchor))
                                if key in controls}, **variants}
    final, counts = evaluate(ranges["evaluation"], policies)
    assert all([row["case"] for row in rows] == list(ranges["evaluation"]) for rows in final.values())
    assert [r["actions"][:1] for r in final["CQL_after_refusal"]] == [r["actions"][:1] for r in final[learned_anchor]]
    divergence = {}
    for name in variants:
        first_changes = (next((i for i, (a, b) in enumerate(zip(row["actions"], base_row["actions"])) if a != b), None)
                         for row, base_row in zip(final[name], final[learned_anchor], strict=True))
        divergence[name] = dict(Counter(i for i in first_changes if i is not None))
    assert set(divergence["CQL_first_offer"]) <= {0} and 0 not in divergence["CQL_after_refusal"]
    audits = {key: {name: gate(final, key, name, anchor, settings) for name in dict.fromkeys((learned_anchor, strongest))}
              for key in variants}
    result = {"results": means(final), "decisions_by_stage": counts, "first_change_vs_matched_by_stage": divergence, "gates": audits,
              "runtime": {"seconds": perf_counter() - started, "python": platform.python_version(), "torch": torch.__version__, "device": "cpu"},
              "pilot_passed": bool(selected and all(item["passed"] for item in audits[selected].values()))}
    write_json(out / "evaluation_rows.json", final)
    write_json(out / "results.json", freeze | result)
    print(json.dumps({"selected": selected, "passed": result["pilot_passed"], "results": result["results"], "runtime": result["runtime"]}, indent=2))
    signal.alarm(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/simbank_timed_override.json"))
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run(args.out, args.config)
