"""Seeding determinism and provenance-manifest writing (rule E, CI-enforced properties)."""

import json
import random

from grounded_matsci.reproducibility import experiment_id, seed, write_manifest

REQUIRED_MANIFEST_FIELDS = {
    "git_sha",
    "git_dirty",
    "config_hash",
    "config_path",
    "dataset_manifest",
    "split_manifest",
    "seed",
    "python_version",
    "package_version",
    "timestamp_utc",
    "hostname",
    "gpu_model",
    "cuda_version",
    "driver_version",
}


def test_seed_is_deterministic():
    seed(123)
    a = [random.random() for _ in range(5)]
    seed(123)
    b = [random.random() for _ in range(5)]
    assert a == b


def test_seed_covers_numpy():
    numpy = __import__("numpy")
    seed(7)
    a = numpy.random.rand(3).tolist()
    seed(7)
    b = numpy.random.rand(3).tolist()
    assert a == b


def test_manifest_has_required_fields(tmp_path):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("seed: 1\n")
    path = write_manifest(tmp_path, config_path=cfg, seed_value=1)
    manifest = json.loads(path.read_text())
    assert set(manifest) == REQUIRED_MANIFEST_FIELDS
    assert manifest["seed"] == 1
    assert manifest["config_hash"].startswith("sha256:")


def test_experiment_id_format():
    from datetime import UTC, datetime

    eid = experiment_id("abcdef1234", "sha256:0123456789", datetime(2026, 7, 7, tzinfo=UTC))
    assert eid == "abcdef12_01234567_20260707T000000Z"
