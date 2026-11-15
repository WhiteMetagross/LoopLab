"""Run the complete LoopLab experiment chain."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

from src.runtime import enable_shared_runtime


ROOT = Path(__file__).resolve().parents[1]


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def command(module: str, *arguments: object):
    invocation = [sys.executable, "-m", module, *map(str, arguments)]
    print("RUN:", " ".join(invocation), flush=True)
    subprocess.run(invocation, cwd=ROOT, check=True)


def run_suite(config: Path, mode: str, output: Path, workers: int, threads: int):
    arguments = ["--config", config, "--mode", mode, "--out", output,
                 "--workers", workers, "--threads", threads, "--device", "cpu"]
    if (output / "manifest.json").exists():
        arguments.append("--resume")
    command("scripts.run_paper_suite", *arguments)


def resolved_configs(mode: str, root: Path):
    config_dir = root / "resolved_configs"
    common = deepcopy(read(ROOT / "configs/paper_suite.json"))
    focused = deepcopy(read(ROOT / "configs/focused_algorithm_study.json"))
    refined = deepcopy(read(ROOT / "configs/focused_refinement_study.json"))
    ablations = deepcopy(read(ROOT / "configs/regression_ablation_suite.json"))

    focused["selection"]["discovery_run"] = str(root / "common")
    refined["selection"]["source_main_run"] = str(root / "focused")
    ablations["focused_config"] = str(config_dir / "refined.json")
    ablations["paper_config"] = str(config_dir / "refined.json")
    ablations["sources"] = {
        "insurance": str(root / "focused/insurance"),
        "bpisim": str(root / "refined/bpisim"),
        "manufacturing": str(root / "refined/manufacturing"),
        "simbank": str(root / "refined/simbank"),
    }
    paths = {name: config_dir / f"{name}.json" for name in ("common", "focused", "refined", "ablations")}
    for path, value in zip(paths.values(), (common, focused, refined, ablations), strict=True):
        write(path, value)
    return paths


def regression_device(requested: str):
    if requested != "auto":
        return requested
    enable_shared_runtime()
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "full"), default="full")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--regression-device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--skip-data", action="store_true")
    args = parser.parse_args()

    os.environ.update(OMP_NUM_THREADS=str(args.threads), OPENBLAS_NUM_THREADS=str(args.threads),
                      MKL_NUM_THREADS=str(args.threads), NUMEXPR_NUM_THREADS=str(args.threads))
    if not args.skip_data:
        command("scripts.prepare_data")

    command("unittest", "discover", "-s", "tests", "-p", "test_*.py")
    output = ROOT / f"runs/reproduction/{args.mode}"
    configs = resolved_configs(args.mode, output)

    run_suite(configs["common"], args.mode, output / "common", args.workers, args.threads)
    run_suite(configs["focused"], args.mode, output / "focused", args.workers, args.threads)
    if args.mode == "full":
        command("scripts.summarize_focused_algorithm_study", "--run", output / "focused",
                "--config", configs["focused"])
    run_suite(configs["refined"], args.mode, output / "refined", args.workers, args.threads)
    if args.mode == "full":
        command("scripts.summarize_focused_refinement", "--run", output / "refined",
                "--config", configs["refined"])

    device = regression_device(args.regression_device)
    regression_workers = min(args.workers, 3) if device == "cuda" else args.workers
    ablation_output = output / "ablations"
    audit_path = ablation_output / "audit.json"
    completed_audit = read(audit_path) if audit_path.exists() else None
    if not completed_audit or completed_audit.get("status") != "pass" or completed_audit.get("mode") != args.mode:
        command("scripts.run_regression_ablations", "--config", configs["ablations"], "--mode", args.mode,
                "--out", ablation_output, "--workers", regression_workers,
                "--threads", args.threads, "--device", device)
        command("scripts.audit_regression_ablations", "--run", ablation_output)
    command("scripts.summarize_regression_ablations", "--run", output / "ablations")

    if args.mode == "full":
        command("scripts.build_common_artifacts", "--run", output / "common", "--paper", ROOT / "paper",
                "--report", ROOT / "report/paper_figures")
        command("scripts.build_regression_paper_artifacts", "--run", output / "ablations",
                "--paper", ROOT / "paper", "--report", ROOT / "report/paper_figures")
    print(json.dumps({"status": "complete", "mode": args.mode, "output": str(output),
                      "regression_device": device}, indent=2))


if __name__ == "__main__":
    main()
