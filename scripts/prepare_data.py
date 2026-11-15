"""Acquire the three event logs and the pinned SimBank dependency."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

from src.domain_loop_data import acquire, sha256


ROOT = Path(__file__).resolve().parents[1]


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def prepare_logs() -> dict[str, str]:
    config = read(ROOT / "configs/domain_loop_simulations.json")
    checked = {}
    for domain, spec in config["domains"].items():
        path = acquire(spec, config["maximum_download_bytes"], download=True)
        checked[domain] = sha256(path.read_bytes())
    return checked


def prepare_simbank() -> dict[str, str]:
    release = read(ROOT / "configs/simbank_release.json")
    checkout = ROOT / "external/SimBank"
    if not checkout.exists():
        checkout.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "--branch", release["tag"], "--single-branch",
             release["repository"], str(checkout)], check=True)
    commit = subprocess.check_output(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True
    ).strip()
    if commit != release["commit"]:
        raise ValueError(f"SimBank checkout differs from the pinned commit: {commit}")

    record = release["parameter_file"]
    spec = {
        "source_path": str(ROOT / "data/simbank/released" / record["name"]),
        "source_url": record["url"],
        "sha256": record["sha256"],
        "maximum_download_bytes": 1_000_000,
    }
    path = acquire(spec, spec["maximum_download_bytes"], download=True)
    return {"commit": commit, "parameter_sha256": sha256(path.read_bytes())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    report = {"event_logs": prepare_logs(), "simbank": prepare_simbank()}
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
