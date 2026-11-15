"""Shared Torch environment discovery for the WSL Conda layout."""

from pathlib import Path
import sys


def enable_shared_runtime(env_name="trainenv"):
    path = (Path(sys.prefix).parent / env_name / "lib" /
            f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages")
    if path.exists() and str(path) not in sys.path:
        sys.path.append(str(path))
    return str(path)
