"""Run the Regression focused ablation suite."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from importlib.metadata import version
import json
import multiprocessing
import os
from pathlib import Path
import platform
import random
import subprocess
import time
import traceback

import numpy as np
from threadpoolctl import threadpool_limits

from scripts import regression_routing_ablations, regression_simbank_ablations
from scripts.summarize_focused_algorithm_study import hierarchical_interval
from src.benchmark_models import seeds
from src.domain_loop_data import sha256
from src.domain_loop_simulations import save
from src.runtime import enable_shared_runtime


DOMAINS = ("insurance", "bpisim", "manufacturing", "simbank")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def status(path, stage, **extra):
    save(path, {"stage": stage, "utc": datetime.now(timezone.utc).isoformat(), "pid": os.getpid(), **extra})


def run_job(out, domain, index, seed, config, profile, device, threads):
    folder = out / domain / f"seed_{index}"
    folder.mkdir(parents=True)
    state = out / domain / f"seed_{index}.status.json"
    started = time.perf_counter()

    def progress(stage):
        status(state, stage, seed=seed, seconds=time.perf_counter() - started)

    try:
        random.seed(seed)
        np.random.seed(seed)
        progress("running")
        with threadpool_limits(limits=threads):
            if domain == "simbank":
                enable_shared_runtime()
                regression_simbank_ablations.run(folder, index, seed, config, profile, progress, device)
            else:
                regression_routing_ablations.run(folder, index, seed, domain, config, profile, progress)
        save(folder / "complete.json", {"seed": seed, "seconds": time.perf_counter() - started})
        progress("complete")
        print(f"Completed {domain} seed {index} in {time.perf_counter() - started:.1f}s", flush=True)
    except Exception:
        (folder / "error.log").write_text(traceback.format_exc(), encoding="utf-8")
        progress("failed")
        raise


def summarize_domain(out, domain, fitting_seeds, draws, master):
    folders = [out / domain / f"seed_{i}" for i in range(len(fitting_seeds))]
    designs = [read(folder / "design.json") for folder in folders]
    fields, contrasts = designs[0]["fields"], designs[0]["contrasts"]
    if any(row["fields"] != fields or row["contrasts"] != contrasts for row in designs[1:]):
        raise ValueError(f"{domain} changed ablation schema across seeds")
    panels = []
    for folder in folders:
        with np.load(folder / "paired.npz") as data:
            panels.append({name: data[name] for name in data.files})

    def named(values):
        return dict(zip(fields, np.asarray(values).tolist(), strict=True))

    policies = {}
    common = set.intersection(*(set(panel) for panel in panels))
    for name in sorted(common):
        means = np.asarray([panel[name].mean(axis=0) for panel in panels])
        policies[name] = {"mean": named(means.mean(axis=0)),
                          "seed_sd": named(means.std(axis=0, ddof=1)) if len(means) > 1 else None,
                          "by_seed": {str(seed): named(value) for seed, value in zip(fitting_seeds, means, strict=True)}}
    comparisons = {}
    rng = np.random.default_rng(np.random.SeedSequence([master, 977, DOMAINS.index(domain)]))
    for label, terms in contrasts.items():
        if any(not terms.keys() <= panel.keys() for panel in panels):
            comparisons[label] = {"status": "missing_arm"}
            continue
        values = [sum(weight * panel[name] for name, weight in terms.items()) for panel in panels]
        means = np.asarray([value.mean(axis=0) for value in values])
        interval = hierarchical_interval(values, rng, draws, case_level=domain == "simbank")
        comparisons[label] = {"mean": named(means.mean(axis=0)),
                              "paired_ci95": named(interval[[0, 2]].T),
                              "seed_sd": named(means.std(axis=0, ddof=1)) if len(means) > 1 else None,
                              "by_seed": {str(seed): named(value) for seed, value in zip(fitting_seeds, means, strict=True)}}
    return {"fields": fields, "policies": policies, "comparisons": comparisons,
            "design_by_seed": dict(zip(map(str, fitting_seeds), designs, strict=True))}


def validate_config(config, check_sources=True):
    if config["master_seed"] != 17 or config["replicates"] != 3:
        raise ValueError("Ablation suite requires master seed 17 and three random seeds")
    if config["smoke"]["namespace"] == config["full"]["namespace"]:
        raise ValueError("Smoke and full namespaces must differ")
    if config["routing"]["forecast_families"] != ["Regression"]:
        raise ValueError("Ablations must hold the Regression family fixed")
    source_roots = {Path(value) for value in config["sources"].values()}
    if check_sources and any(not root.is_dir() for root in source_roots):
        raise ValueError("A frozen focused source run is missing")


def preflight(config_path, mode):
    config = read(config_path)
    validate_config(config)
    fitting_seeds = seeds(config["master_seed"], config["replicates"])
    if mode == "smoke":
        fitting_seeds = fitting_seeds[:1]
    sources = {}
    for domain, root in config["sources"].items():
        sources[domain] = []
        for index, seed in enumerate(fitting_seeds):
            folder = Path(root) / f"seed_{index}"
            complete, freeze = read(folder / "complete.json"), read(folder / "freeze.json")
            if complete["seed"] != seed or freeze["forecast_family"] != "Regression":
                raise ValueError(f"Invalid frozen source for {domain} seed {index}")
            sources[domain].append({"seed": seed, "folder": str(folder)})
    profile, bank = config[mode], config["bank"]
    ranges = []
    for index, seed in enumerate(fitting_seeds):
        base = bank["scenario_base"] + index * bank["seed_stride"] + profile.get("smoke_offset", 0)
        direct = [(base, base + profile["bank_validation_cases"], f"validation_{index}"),
                  (base + bank["test_offset"], base + bank["test_offset"] + profile["bank_cases"], f"test_{index}")]
        logging_base = bank["scenario_base"] + (index + profile["bank_seed_offset"]) * bank["seed_stride"]
        logging = [(logging_base, logging_base + profile["bank_fit"], f"logging_train_{index}"),
                   (logging_base + bank["test_offset"],
                    logging_base + bank["test_offset"] + profile["bank_cases"], f"logging_test_{index}")]
        for start, stop, label in direct + logging:
            if stop <= start:
                raise ValueError(f"Empty SimBank range: {label}")
            ranges.append((start, stop, label, seed))
    for i, first in enumerate(ranges):
        for second in ranges[i + 1:]:
            if max(first[0], second[0]) < min(first[1], second[1]):
                raise ValueError(f"SimBank ranges overlap: {first[2]} and {second[2]}")
    return {"status": "pass", "mode": mode, "seeds": fitting_seeds,
            "jobs": len(DOMAINS) * len(fitting_seeds), "sources": sources,
            "routing_namespace": profile["namespace"],
            "routing_test_workloads_per_seed": profile["test_workloads"],
            "simbank_test_cases_per_seed": profile["bank_cases"],
            "simbank_ranges": [{"start": a, "stop": b, "label": label, "seed": seed}
                               for a, b, label, seed in ranges]}


def run(out, config_path, mode, workers, threads, device):
    started = time.perf_counter()
    config = read(config_path)
    validate_config(config)
    profile = config[mode]
    fitting_seeds = seeds(config["master_seed"], config["replicates"])
    if mode == "smoke":
        fitting_seeds = fitting_seeds[:1]
    enable_shared_runtime()
    import torch
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    out.mkdir(parents=True)
    sources = [config_path, Path(config["domain_config"]), Path(config["focused_config"]),
               Path(config["paper_config"]), Path("scripts/regression_routing_ablations.py"),
               Path("scripts/regression_simbank_ablations.py"), Path("scripts/run_regression_ablations.py"),
               Path("scripts/simbank_logging_followup.py"), Path("src/paper_routing.py")]
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                         capture_output=True, text=True, check=False).stdout.strip() if device == "cuda" else None
    save(out / "manifest.json", {"mode": mode, "config": config, "seeds": fitting_seeds,
         "runtime": {"python": platform.python_version(), "torch": torch.__version__,
                     "torch_cuda": torch.version.cuda, "device": device, "gpu": gpu,
                     "versions": {name: version(name) for name in
                                  ("numpy", "scipy", "scikit-learn", "catboost", "xgboost-cpu")}},
         "source_sha256": {str(path): sha256(path.read_bytes()) for path in sources}})
    status(out / "status.json", "running", total_jobs=len(DOMAINS) * len(fitting_seeds))
    failures = {}
    context = multiprocessing.get_context("spawn")
    bank_workers = 1 if device == "cuda" else min(workers, len(fitting_seeds))
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as cpu, \
         ProcessPoolExecutor(max_workers=bank_workers, mp_context=context) as bank:
        futures = {}
        for domain in DOMAINS:
            executor = bank if domain == "simbank" else cpu
            for index, seed in enumerate(fitting_seeds):
                future = executor.submit(run_job, out, domain, index, seed, config, profile, device, threads)
                futures[future] = (domain, index)
        for future in as_completed(futures):
            domain, index = futures[future]
            try:
                future.result()
            except Exception as exc:
                failures[f"{domain}_seed_{index}"] = str(exc)
    domains = {domain: summarize_domain(out, domain, fitting_seeds, profile["bootstrap_draws"], config["master_seed"])
               for domain in DOMAINS if not any(key.startswith(f"{domain}_") for key in failures)}
    save(out / "results.json", {"mode": mode, "seeds": fitting_seeds,
         "seconds": time.perf_counter() - started, "failures": failures, "domains": domains})
    status(out / "status.json", "failed" if failures else "complete", failures=failures,
           seconds=time.perf_counter() - started)
    if failures:
        raise RuntimeError(f"Regression ablation jobs failed: {failures}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "full"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/regression_ablation_suite.json"))
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if args.preflight_only:
        print(json.dumps(preflight(args.config, args.mode), indent=2))
    else:
        run(args.out, args.config, args.mode, args.workers, args.threads, args.device)
