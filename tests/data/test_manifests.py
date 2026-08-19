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


def test_frozen_inputs_manifest():
    _check_manifest(ROOT / "data" / "manifests" / "frozen_inputs.yaml")


def test_frozen_dirs_fully_manifested():
    manifest = yaml.safe_load((ROOT / "data" / "manifests" / "frozen_inputs.yaml").read_text())
    listed = {e["path"] for e in manifest["files"]}
    on_disk = {
        str(p.relative_to(ROOT))
        for d in ("frozen", "pilot")
        for p in (ROOT / "data" / d).iterdir()
        if p.is_file()
    }
    assert listed == on_disk


def test_results_dir_fully_manifested():
    manifest = yaml.safe_load((ROOT / "data" / "manifests" / "results_manifest.yaml").read_text())
    listed = {e["path"] for e in manifest["files"]}
    on_disk = {str(p.relative_to(ROOT)) for p in (ROOT / "results").rglob("*") if p.is_file()}
    assert listed == on_disk


def test_traces_manifest():
    _check_manifest(ROOT / "data" / "manifests" / "traces.yaml")


def test_traces_dir_fully_manifested():
    manifest = yaml.safe_load((ROOT / "data" / "manifests" / "traces.yaml").read_text())
    listed = {e["path"] for e in manifest["files"]}
    on_disk = {
        str(p.relative_to(ROOT)) for p in (ROOT / "data" / "traces").iterdir() if p.is_file()
    }
    assert listed == on_disk


def test_pending_traces_manifest_lists_missing_files_only():
    manifest = yaml.safe_load((ROOT / "data" / "manifests" / "pending_traces.yaml").read_text())
    assert manifest["status"] == "not_included"
    assert manifest["files"], "pending-traces manifest must enumerate the missing files"
