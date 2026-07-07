"""Hash verification of committed data and frozen results against their manifests
(rule F.1: a loader must verify hashes before use)."""

import hashlib
from pathlib import Path

import yaml

ROOT = Path(__file__).parent.parent.parent


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _check_manifest(manifest_path: Path) -> None:
    manifest = yaml.safe_load(manifest_path.read_text())
    for entry in manifest["files"]:
        p = ROOT / entry["path"]
        assert p.exists(), f"{entry['path']} listed in {manifest_path.name} but missing"
        assert _sha256(p) == entry["sha256"], f"hash mismatch for {entry['path']}"


def test_isotope_gt_manifest():
    _check_manifest(ROOT / "data" / "manifests" / "isotope_gt.yaml")


def test_results_manifest():
    _check_manifest(ROOT / "data" / "manifests" / "results_manifest.yaml")


def test_handoff_manifest_lists_missing_files_only():
    manifest = yaml.safe_load((ROOT / "data" / "manifests" / "handoff_missing.yaml").read_text())
    assert manifest["status"] == "not_included"
    assert manifest["files"], "handoff manifest must enumerate the missing files"
