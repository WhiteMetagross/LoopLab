"""Fit, check, or generate trajectories from SimClaims, SimFactory, and BPISim."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import sys
from time import perf_counter

import numpy as np

from src.domain_loop_data import DOMAIN_CHOICES, acquire, chronological_split, load_cases, selected_domains, sha256
from src.domain_loop_simulator import ACTIONS, LoopEnvironment, calibrate, run_policy


def metrics(rows):
    n = len(rows)
    outcomes = Counter(row["outcome"] for row in rows)
    elapsed = [row["elapsed"] for row in rows]
    return {"cases": n, "outcomes": dict(outcomes),
            "repeat_cases": sum(row["attempts"] > 1 for row in rows),
            "repeat_rate": sum(row["attempts"] > 1 for row in rows) / n,
            "extra_attempts_per_case": sum(row["attempts"] - 1 for row in rows) / n,
            "resolution_rate": outcomes["resolved"] / n,
            "negative_exit_rate": outcomes["negative"] / n,
            "unobserved_exit_rate": outcomes["unobserved"] / n,
            "cap_rate": outcomes["cap"] / n,
            "mean_elapsed_minutes": float(np.mean(elapsed)),
            "median_elapsed_minutes": float(np.median(elapsed)),
            "p90_elapsed_minutes": float(np.quantile(elapsed, .9)),
            "mean_work_minutes": sum(row["work"] for row in rows) / n,
            "mean_pre_service_minutes": sum(row["pre_service_wait"] for row in rows) / n,
            "mean_queue_minutes": sum(row["queue_wait"] for row in rows) / n,
            "specialist_attempts_per_case": sum(row["specialist_attempts"] for row in rows) / n}


def observed_metrics(cases):
    rows = [{"attempts": len(case.attempts), "outcome": case.attempts[-1].outcome,
             "elapsed": sum(item.elapsed for item in case.attempts), "work": sum(item.service for item in case.attempts),
             "pre_service_wait": sum(item.before_service or 0. for item in case.attempts),
             "queue_wait": 0., "specialist_attempts": 0} for case in cases]
    result = metrics(rows)
    # Source queues cannot be separated from observed elapsed gaps.
    result.pop("mean_queue_minutes")
    result.pop("specialist_attempts_per_case")
    result["attempt_histogram"] = dict(Counter(row["attempts"] for row in rows))
    return result


def historical_check(laws, holdout, spec, cfg, seeds):
    observed = observed_metrics(holdout)
    outcomes = [len(case.attempts) > 1 for case in holdout]
    predictions = {}
    key_metrics = ("repeat_rate", "extra_attempts_per_case", "resolution_rate", "negative_exit_rate",
                   "unobserved_exit_rate", "cap_rate", "median_elapsed_minutes", "p90_elapsed_minutes",
                   "mean_pre_service_minutes", "mean_work_minutes")
    for name, law in laws.items():
        env = LoopEnvironment(law, spec, cfg, unlimited=True)
        all_metrics = [metrics(run_policy(env, [0.] * len(holdout), seed, lambda _: "standard")) for seed in seeds]
        simulated = {}
        for key in key_metrics:
            values = [row[key] for row in all_metrics]
            simulated[key] = {"mean": float(np.mean(values)), "p05": float(np.quantile(values, .05)),
                              "p95": float(np.quantile(values, .95))}
        p = law.probability(1)[0]
        predictions[name] = {"first_attempt_brier": float(np.mean((p - np.array(outcomes)) ** 2)),
                             "simulated": simulated, "model": law.summary()}
    return {"observed": observed, "models": predictions}


def queue_rule(env):
    extra_service = (env.multiplier - 1) * env.law.mean_service
    expected_next = env.law.expected_elapsed(2, env.scenario_repeat)

    def pick(state):
        saved = state.repeat_probability * env.effect * expected_next
        extra = extra_service + state.specialist_wait_estimate - state.standard_wait_estimate
        return "specialist" if saved > extra else "standard"

    return pick


def arrival_minutes(rng, law, spec, count, load):
    average_gap = law.mean_service / (spec["standard_servers"] * load)
    return np.r_[0., rng.exponential(average_gap, count - 1).cumsum()]


def interventions(law, spec, cfg, seed):
    rng = np.random.default_rng(seed)
    result = []
    for load in cfg["loads"]:
        workloads = [(arrival_minutes(rng, law, spec, cfg["cases_per_workload"], load), int(rng.integers(2**63)))
                     for _ in range(cfg["policy_workloads"])]
        for effect in cfg["specialist_effects"]:
            env = LoopEnvironment(law, spec, cfg, effect, scenario=True)
            policies = {"standard_only": lambda _: "standard", "always_specialist": lambda _: "specialist",
                        "escalate_after_repeat": lambda state: "specialist" if state.attempt > 1 else "standard",
                        "queue_aware": queue_rule(env)}
            baseline = []
            for name, policy in policies.items():
                rows = [run_policy(env, arrivals, noise, policy) for arrivals, noise in workloads]
                if name == "standard_only":
                    baseline = rows
                paired = []
                for trial, (reference, candidate) in enumerate(zip(baseline, rows)):
                    if effect == 0:
                        assert all((a["attempts"], a["outcome"]) == (b["attempts"], b["outcome"])
                                   for a, b in zip(reference, candidate))
                    a, b = metrics(reference), metrics(candidate)
                    paired.append({"workload": trial, "metrics": b,
                                   "change_in_repeat_rate": b["repeat_rate"] - a["repeat_rate"],
                                   "change_in_resolution_rate": b["resolution_rate"] - a["resolution_rate"],
                                   "change_in_mean_elapsed_minutes": b["mean_elapsed_minutes"] - a["mean_elapsed_minutes"]})
                result.append({"load": load, "specialist_effect": effect, "policy": name,
                               "aggregate": metrics([item for batch in rows for item in batch]), "paired": paired})
    return result


def generate(law, spec, cfg, seed, path, load, effect, workloads, cases, probability, route_when_ready=False):
    """Emit only synthetic event records and online decision transitions."""
    workload_seed, behavior_seed = seed.spawn(2)
    rng = np.random.default_rng(workload_seed)
    behavior = np.random.default_rng(behavior_seed)
    env = LoopEnvironment(law, spec, cfg, effect, scenario=True, route_when_ready=route_when_ready)
    totals = []
    with (path / "events.jsonl").open("x", encoding="utf-8") as events, (path / "transitions.jsonl").open("x", encoding="utf-8") as transitions:
        for workload in range(workloads):
            arrivals = arrival_minutes(rng, law, spec, cases, load)
            state = env.reset(arrivals, int(rng.integers(2**63)))
            while state is not None:
                action = ACTIONS[int(behavior.random() < probability)]
                next_state, reward, done, completed = env.step(action)
                record = {"workload": workload, "state": asdict(state), "action": action,
                          "propensity": probability if action == "specialist" else 1 - probability,
                          "reward": reward, "done": done, "next_state": asdict(next_state) if next_state else None,
                          "completed": [asdict(row) for row in completed]}
                transitions.write(json.dumps(record) + "\n")
                state = next_state
            for event in env.event_log():
                events.write(json.dumps({"workload": workload, **event}) + "\n")
            totals.extend(env.case_rows())
    return metrics(totals)


def save(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/domain_loop_simulations.json"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--domain", type=str.lower, choices=DOMAIN_CHOICES, default="both")
    parser.add_argument("--download", action="store_true", help="Download missing checksum pinned input files")
    parser.add_argument("--mode", choices=("evaluate", "generate"), default="evaluate")
    parser.add_argument("--seed-index", type=int, default=0, help="Index of a child of the configured master seed")
    parser.add_argument("--workloads", type=int, default=4, help="Generator workloads per domain")
    parser.add_argument("--cases", type=int, default=80, help="Generator cases per workload")
    parser.add_argument("--load", type=float, default=.6, help="Generator arrival load")
    parser.add_argument("--effect", type=float, default=.25, help="Generator specialist effect")
    parser.add_argument("--probability", type=float, default=.5, help="Generator probability of specialist routing")
    args = parser.parse_args()
    if not (0 <= args.seed_index < 10000 and args.workloads > 0 and args.cases > 0
            and 0 < args.probability < 1 and np.isfinite(args.load) and args.load > 0 and 0 <= args.effect <= 1):
        parser.error("Invalid seed index, generator workload, load, propensity, or specialist effect")
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    if not (0 < cfg["fit_fraction"] < 1 and cfg["stage_prior_cases"] > 0 and cfg["minimum_timing_cell"] > 0):
        raise ValueError("Invalid calibration settings")
    root_seed = np.random.SeedSequence(cfg["master_seed"]).spawn(args.seed_index + 1)[-1]
    seed = int(root_seed.generate_state(1)[0] % (2**31 - 1))
    args.out.mkdir(parents=True, exist_ok=False)
    start = perf_counter()
    module_dir = Path(__file__).parent
    provenance = {"seed": seed, "seed_index": args.seed_index, "arguments": vars(args) | {"config": str(args.config), "out": str(args.out)},
                  "config": cfg, "started_utc": datetime.now(timezone.utc).isoformat(),
                  "runtime": {"python": sys.version.split()[0], "numpy": np.__version__, "platform": platform.platform(), "device": "CPU"},
                  "code_sha256": {name: sha256((module_dir / name).read_bytes()) for name in
                                  ("domain_loop_simulations.py", "domain_loop_simulator.py", "domain_loop_data.py",
                                   "bpisim_data.py", "bpi_queue.py", "cycle_data.py", "loop_feasibility.py")}}
    save(args.out / "manifest.json", provenance)

    for index, domain, spec in selected_domains(cfg, args.domain):
        domain_start = perf_counter()
        path = args.out / domain
        path.mkdir()
        source = acquire(spec, cfg["maximum_download_bytes"], args.download)
        cases, audit = load_cases(source, domain, spec)
        fit, holdout, boundary, excluded = chronological_split(cases, cfg["fit_fraction"])
        law = calibrate(fit, spec, cfg)
        name = spec.get("name", domain)
        result = {"domain": domain, "simulator": name, "source": spec, "data_audit": audit, "fit": observed_metrics(fit),
                  "split": {"boundary_utc_or_naive_as_utc": datetime.fromtimestamp(boundary * 60, timezone.utc).isoformat(),
                            "fit_cases": len(fit), "holdout_cases": len(holdout), "overlapping_cases_excluded": excluded,
                            "fit_case_key_sha256": sha256(json.dumps(sorted(case.key for case in fit)).encode()),
                            "holdout_case_key_sha256": sha256(json.dumps(sorted(case.key for case in holdout)).encode())},
                  "calibration": law.summary()}
        print(f"{name}: {audit['source_cases']} source cases, {len(fit)} fit, {len(holdout)} holdout", flush=True)
        if args.mode == "evaluate":
            rng = np.random.default_rng(np.random.SeedSequence([seed, index, 0]))
            pooled = calibrate(fit, spec, cfg, by_attempt=False)
            draws = [int(value) for value in rng.integers(2**63, size=cfg["historical_replications"])]
            result["fidelity"] = historical_check({"pooled": pooled, "by_attempt": law}, holdout, spec, cfg, draws)
            result["interventions"] = interventions(law, spec, cfg, np.random.SeedSequence([seed, index, 1]))
        else:
            result["generated"] = generate(law, spec, cfg, np.random.SeedSequence([seed, index, 2]), path,
                                           args.load, args.effect, args.workloads, args.cases, args.probability)
        result["elapsed_seconds"] = perf_counter() - domain_start
        save(path / "results.json", result)
    provenance["elapsed_seconds"] = perf_counter() - start
    provenance["status"] = "complete"
    save(args.out / "manifest.json", provenance)


if __name__ == "__main__":
    main()
