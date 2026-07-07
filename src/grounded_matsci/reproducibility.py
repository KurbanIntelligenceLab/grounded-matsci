"""Seeding and per-run provenance manifests (rule E).

Every CLI entrypoint calls `seed()` once near startup and `start_run()` to create
`outputs/<experiment_id>/` with a config snapshot and a `manifest.json`.

Note on scope: this project uses no PyTorch and no GPU; the GPU/CUDA manifest fields are
recorded as None. The experiment runners' nondeterminism is dominated by live LLM API
sampling, which seeding cannot control — the seed covers `random`, NumPy, and
PYTHONHASHSEED for the deterministic analysis paths.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import shutil
import socket
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path


def seed(value: int) -> None:
    """Seed all RNG sources used by this package. Call exactly once near startup."""
    os.environ["PYTHONHASHSEED"] = str(value)
    random.seed(value)
    try:
        import numpy

        numpy.random.seed(value)
    except ImportError:
        pass


def _git_info() -> tuple[str, bool]:
    """(git_sha, git_dirty) of the working tree, or ("unknown", True) outside a repo."""
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
            ).stdout.strip()
        )
        return sha, dirty
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown", True


def config_hash(config_path: str | Path) -> str:
    return "sha256:" + hashlib.sha256(Path(config_path).read_bytes()).hexdigest()


def experiment_id(git_sha: str, cfg_hash: str, now: datetime) -> str:
    """`<short_git_sha>_<config_hash_prefix>_<utc_timestamp>` (rule E.2)."""
    return f"{git_sha[:8]}_{cfg_hash.removeprefix('sha256:')[:8]}_{now:%Y%m%dT%H%M%SZ}"


def write_manifest(
    output_dir: str | Path,
    *,
    config_path: str | Path,
    seed_value: int,
    dataset_manifest: str | None = None,
    split_manifest: str | None = None,
) -> Path:
    """Write the per-experiment `manifest.json` (rule E.2). Returns its path."""
    git_sha, git_dirty = _git_info()
    try:
        from grounded_matsci import __version__ as package_version
    except ImportError:
        package_version = "unknown"
    manifest = {
        "git_sha": git_sha,
        "git_dirty": git_dirty,
        "config_hash": config_hash(config_path),
        "config_path": str(config_path),
        "dataset_manifest": dataset_manifest,
        "split_manifest": split_manifest,
        "seed": seed_value,
        "python_version": platform.python_version(),
        "package_version": package_version,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "hostname": socket.gethostname(),
        "gpu_model": None,
        "cuda_version": None,
        "driver_version": None,
    }
    out = Path(output_dir) / "manifest.json"
    out.write_text(json.dumps(manifest, indent=2) + "\n")
    return out


def start_run(
    config_path: str | Path,
    seed_value: int,
    outputs_root: str | Path = "outputs",
    dataset_manifest: str | None = None,
    split_manifest: str | None = None,
    debug: bool = False,
) -> Path:
    """Create `outputs/<experiment_id>/`, snapshot the config, write the manifest.

    Final-result runs must come from a clean tree (rule E.2): raises if the git tree is
    dirty unless `debug=True`.
    """
    git_sha, git_dirty = _git_info()
    if git_dirty and not debug:
        raise RuntimeError(
            "git tree is dirty; final-result runs must be clean (pass --debug to override)"
        )
    cfg_hash = config_hash(config_path)
    exp_id = experiment_id(git_sha, cfg_hash, datetime.now(UTC))
    run_dir = Path(outputs_root) / exp_id
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(config_path, run_dir / "config_snapshot.yaml")
    write_manifest(
        run_dir,
        config_path=config_path,
        seed_value=seed_value,
        dataset_manifest=dataset_manifest,
        split_manifest=split_manifest,
    )
    if sys.stdout.isatty():
        print(f"run dir: {run_dir}")
    return run_dir
