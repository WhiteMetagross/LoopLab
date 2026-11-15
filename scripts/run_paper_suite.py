"""Run, resume, and summarize the four simulator paper experiments."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import fcntl
from importlib.metadata import version
import json
import multiprocessing
import os
from pathlib import Path
import platform
import random
import time
import traceback

import numpy as np
from threadpoolctl import threadpool_limits

from scripts import specialist_slot_search
from scripts.audit_routing_rewards import gaps
from src import paper_rl, paper_routing, paper_simbank
from src.benchmark_models import seeds
from src.domain_loop_data import sha256
from src.domain_loop_simulations import save
from src.runtime import enable_shared_runtime


DOMAINS = ("simbank", "insurance", "bpisim", "manufacturing")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def utc():
    return datetime.now(timezone.utc).isoformat()


def run_job(job, out, config, mode, threads, device):
    name, seed_index, seed = job
    target = out / name / f"seed_{seed_index}"
    state = target.parent / f"seed_{seed_index}.status.json"
    start = time.perf_counter()
    stages = {}

    def progress(stage):
        stages[stage] = time.perf_counter() - start
        save(state, {"task": name, "seed": seed, "stage": stage, "utc": utc(), "pid": os.getpid()})

    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        progress("starting")
        random.seed(seed)
        np.random.seed(seed)
        budget = config[mode]
        with threadpool_limits(limits=threads):
            if name in config["variants"]:
                domain, _ = config["variants"][name]
                resolved = out / "resolved_variants" / f"{name}.json"
                progress("retained_policy_search")
                specialist_slot_search.run(target, resolved, seed_index, config["variant_namespace"][mode], domain)
            else:
                target.mkdir()
                enable_shared_runtime()
                import torch
                torch.set_num_threads(threads)
                torch.manual_seed(seed)
                if device == "cuda" and not torch.cuda.is_available():
                    raise RuntimeError("CUDA requested but unavailable")
                paper_rl.DEVICE = device
                save(target / "runtime.json", {"python": platform.python_version(), "torch": torch.__version__,
                     "compiled_cuda": torch.version.cuda, "device": device, "threads": threads,
                     "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
                     "versions": {name: version(name) for name in ("numpy", "catboost", "xgboost-cpu", "scikit-learn", "joblib")}})
                kind = "simbank" if name == "simbank" else "routing"
                settings = config["learners"] | budget["learners"] | config[kind] | budget[kind]
                if kind == "simbank":
                    paper_simbank.run(target, Path(config["simbank_config"]), settings, config["master_seed"], seed, seed_index,
                                      mode, threads, torch, progress)
                else:
                    paper_routing.run(target, read(Path(config["domain_config"])), settings, name, seed,
                                      config["routing_namespace"][mode], threads, torch, progress)
        result = read(target / "results.json")
        if name in DOMAINS:
            policies = set(next(iter(result["metrics"].values())))
            required = {"FittedQ", "TabularQ", "PPO", *(f"CQL_{a:g}" for a in config["learners"]["cql_alphas"])}
            if name != "simbank" and config["routing"].get("episodic_search", {}).get("enabled", False):
                required.add("EpisodicSearch")
            if name == "simbank" and config["simbank"].get("profit_fitted_ensemble_size", 0):
                required.add("FittedQ_profit_ensemble")
            if not required.issubset(policies):
                raise ValueError(f"Missing methods: {required - policies}")
        save(target / "complete.json", {"seed": seed, "seconds": time.perf_counter() - start, "utc": utc(), "stage_seconds": stages})
        progress("complete")
        print(f"Completed {name} seed {seed_index} in {time.perf_counter() - start:.1f}s", flush=True)
        return None
    except Exception:
        target.mkdir(exist_ok=True)
        (target / "error.log").write_text(traceback.format_exc(), encoding="utf-8")
        progress("failed")
        print(f"Failed {name} seed {seed_index}: {target / 'error.log'}", flush=True)
        return str(target)


def intervals(panels, rng, repetitions, case_level):
    if case_level:
        # Case-level paired normal approximation inside each fit. Resample fits outside it.
        within = np.asarray([rng.multivariate_normal(panel.mean(axis=0), np.cov(panel, rowvar=False) / len(panel),
                                                      size=repetitions, method="svd") for panel in panels])
    else:
        within = np.asarray([panel[rng.integers(len(panel), size=(repetitions, len(panel)))].mean(axis=1) for panel in panels])
    seeds = rng.integers(len(panels), size=(repetitions, len(panels)))
    draws = rng.integers(repetitions, size=seeds.shape)
    combined = within[seeds, draws].mean(axis=1)
    return np.quantile(combined, [.025, .95, .975], axis=0)


def paired(root, reports, settings, cell, policies, simbank, rng, repeats):
    fields = reports[0]["fields"]
    contrasts = {}
    files = [np.load(path / ("heldout.npz" if simbank else f"{cell}.npz")) for path in root]
    try:
        for method in sorted(policies):
            for comparison in ("matched_ml", "strongest_feasible_ml"):
                references = [report[comparison] for report in reports]
                actual = [report["selected_cql"] if method == "Selected_CQL" else method for report in reports]
                if any(name is None for name in [*actual, *references]):
                    contrasts[f"{method}_vs_{comparison}"] = {"status": "selection_unavailable", "methods": actual, "references": references}
                    continue
                panels = []
                for rows, report, policy, ref in zip(files, reports, actual, references, strict=True):
                    candidate, baseline = rows[policy], rows[ref]
                    if simbank:
                        anchor, bank = rows[report["profit_reference"]], rows["Bank"]
                        profit = settings["profit_tolerance_fraction"]
                        cancel = settings["cancellation_tolerance_vs_ml"]
                        margins = np.column_stack(((1 - profit) * baseline[:, 0] - candidate[:, 0],
                                                   (1 - profit) * anchor[:, 0] - candidate[:, 0],
                                                   candidate[:, 4] - baseline[:, 4] - cancel,
                                                   candidate[:, 4] - anchor[:, 4] - cancel,
                                                   candidate[:, 4] - bank[:, 4] - settings["cancellation_tolerance_vs_bank"]))
                    else:
                        fifo, limit = rows["FIFO"], settings["limits"]
                        margins = np.column_stack((gaps(candidate, fifo, limit), gaps(candidate, baseline, limit),
                                                   candidate[:, 3] - limit["specialist_share"] * fifo[:, 2]))
                    panels.append(np.column_stack((candidate - baseline, margins)))
                mean_by_seed = np.asarray([panel.mean(axis=0) for panel in panels])
                bounds = intervals(panels, rng, repeats, simbank)
                n = len(fields)
                target = "refusals" if simbank else "extra_attempts"
                index = fields.index(target)
                threshold = settings["minimum_refusal_reduction" if simbank else "minimum_repeat_reduction"]
                checks = {"target_minimum": bool(mean_by_seed[:, index].mean() <= -threshold),
                          "interval_excludes_zero": bool(bounds[2, index] < 0),
                          "all_operating_bounds": bool((bounds[1, n:] <= 0).all())}
                contrasts[f"{method}_vs_{comparison}"] = {"methods": actual, "references": references,
                    "by_seed": [dict(zip(fields, row[:n].tolist(), strict=True)) for row in mean_by_seed],
                    "difference_mean": dict(zip(fields, mean_by_seed[:, :n].mean(axis=0).tolist(), strict=True)),
                    "paired_ci95": dict(zip(fields, bounds[[0, 2], :n].T.tolist(), strict=True)),
                    "operating_margin_upper95": bounds[1, n:].tolist(), "checks": checks, "passed": all(checks.values())}
    finally:
        for file in files:
            file.close()
    return contrasts


def aggregate(out, config, mode, tasks, indices):
    output = {}
    rng = np.random.default_rng(config["master_seed"])
    for name in tasks:
        paths = [out / name / f"seed_{index}" for index in indices]
        reports = [read(path / "results.json") for path in paths]
        if name in config["variants"]:
            if len(indices) == 3:
                specialist_slot_search.summarize_confirmation(out / name, config["variant_namespace"][mode])
            output[name] = {"kind": "retained_improvement", "statuses": [row["passed"] for row in reports]}
            continue
        simbank = name == "simbank"
        settings = config["simbank" if simbank else "routing"]
        output[name] = {}
        for cell in reports[0]["metrics"]:
            table = {}
            for policy, values in reports[0]["metrics"][cell].items():
                table[policy] = {}
                for metric in values:
                    sample = [row["metrics"][cell][policy][metric] for row in reports]
                    table[policy][metric] = {"mean": float(np.mean(sample)), "sd": float(np.std(sample, ddof=1)) if len(sample) > 1 else None,
                                             "by_seed": sample}
            methods = {policy for policy in table if policy.startswith(("TabularQ", "FittedQ", "PPO", "CQL_", "NeuralQ_", "EpisodicSearch"))}
            output[name][cell] = {"policies": table, "paired": paired(paths, reports, settings, cell, methods | {"Selected_CQL"},
                                                                    simbank, rng, 200 if mode == "smoke" else 2500)}
    save(out / "aggregate.json", output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/paper_suite.json"))
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--mode", choices=("smoke", "full"), required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--tasks", nargs="+")
    args = parser.parse_args()
    if min(args.workers, args.threads) < 1 or args.workers * args.threads > (os.cpu_count() or 1):
        parser.error("Invalid worker or thread budget")
    if args.device == "cuda" and args.workers != 1:
        parser.error("Use one worker when sharing the GPU")
    config = read(args.config)
    if config["replicates"] != 3 or config["master_seed"] != read(Path(config["domain_config"]))["master_seed"]:
        parser.error("The study requires three random seeds and one shared master")
    tasks = args.tasks or [*DOMAINS, *config["variants"]]
    if len(tasks) != len(set(tasks)) or set(tasks) - (set(DOMAINS) | set(config["variants"])):
        parser.error("Unknown or duplicate tasks")
    fit_seeds = seeds(config["master_seed"], config["replicates"])
    indices = list(range(1 if args.mode == "smoke" else config["replicates"]))
    jobs = [(name, index, fit_seeds[index]) for index in indices for name in tasks]
    sources = [args.config, Path(config["domain_config"]), Path(config["simbank_config"]),
               *(Path(value[1]) for value in config["variants"].values()), *Path("src").glob("*.py"),
               *(Path("scripts") / name for name in ("run_paper_suite.py", "specialist_slot_search.py", "specialist_slots_pilot.py",
                 "audit_routing_rewards.py", "check_negotiation_randomness.py", "simbank_constrained_reward.py", "simbank_timed_override.py"))]
    manifest = {"mode": args.mode, "config": config, "seeds": fit_seeds, "jobs": [list(job) for job in jobs],
                "workers": args.workers, "threads": args.threads, "device": args.device,
                "sha256": {str(path): sha256(path.read_bytes()) for path in sources}}
    out = args.out
    existing = set(out.iterdir()) if out.exists() else set()
    if not args.resume and existing - {out / ".lock"}:
        parser.error("Output directory is not empty. Use --resume or choose a new path")
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / ".lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if args.resume and read(out / "manifest.json") != manifest:
        parser.error("Resume requires identical configuration, source, and execution settings")
    if not args.resume:
        save(out / "manifest.json", manifest)
        variants = out / "resolved_variants"
        variants.mkdir(exist_ok=True)
        for name, (_, path) in config["variants"].items():
            save(variants / f"{name}.json", read(Path(path)) | config[args.mode]["variants"])
    pending = []
    for job in jobs:
        name, index, _ = job
        directory = out / name / f"seed_{index}"
        if (directory / "complete.json").exists() and (directory / "results.json").exists():
            continue
        if directory.exists():
            archive = out / "attempts" / name
            archive.mkdir(parents=True, exist_ok=True)
            directory.rename(archive / f"seed_{index}_{time.time_ns()}")
        pending.append(job)
    start = time.perf_counter()
    save(out / "status.json", {"stage": "running", "utc": utc(), "total_jobs": len(jobs), "pending": len(pending)})
    errors = []
    finished = len(jobs) - len(pending)
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        future_jobs = {pool.submit(run_job, job, out, config, args.mode, args.threads, args.device): job for job in pending}
        for future in as_completed(future_jobs):
            try:
                error = future.result()
            except Exception:
                name, index, _ = future_jobs[future]
                error = f"{name}/seed_{index}"
                print(traceback.format_exc(), flush=True)
            if error:
                errors.append(error)
            else:
                finished += 1
            save(out / "status.json", {"stage": "running", "utc": utc(), "total_jobs": len(jobs),
                                       "completed": finished, "errors": errors})
    if errors:
        save(out / "status.json", {"stage": "failed", "utc": utc(), "errors": errors})
        raise SystemExit(1)
    save(out / "status.json", {"stage": "aggregating", "utc": utc(), "total_jobs": len(jobs)})
    aggregate(out, config, args.mode, tasks, indices)
    save(out / "status.json", {"stage": "complete", "utc": utc(), "total_jobs": len(jobs), "seconds": time.perf_counter() - start})
    print(f"Completed all {len(jobs)} jobs", flush=True)


if __name__ == "__main__":
    main()
