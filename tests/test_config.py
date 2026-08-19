"""Every committed YAML config must load and validate into its frozen dataclass."""

from pathlib import Path

import pytest

from grounded_matsci import config as C
from grounded_matsci.config import load_config

CONFIG_DIR = Path(__file__).parent.parent / "configs"

PAIRS = [
    ("harness.yaml", C.HarnessConfig),
    ("arms.yaml", C.ArmsConfig),
    ("selfcheck.yaml", C.SelfcheckConfig),
    ("inline_conf.yaml", C.InlineConfConfig),
    ("endtask.yaml", C.EndtaskConfig),
    ("endtask_derived.yaml", C.EndtaskDerivedConfig),
    ("codata.yaml", C.CodataConfig),
    ("two_stage_triage.yaml", C.TwoStageTriageConfig),
    ("gated_constants.yaml", C.GatedConstantsConfig),
    ("small_models.yaml", C.SmallModelsConfig),
    ("isotopes.yaml", C.IsotopesConfig),
    ("pubchem.yaml", C.PubchemConfig),
]


@pytest.mark.parametrize(("name", "cls"), PAIRS)
def test_config_loads(name, cls):
    cfg = load_config(CONFIG_DIR / name, cls)
    assert isinstance(cfg, cls)


def test_every_config_file_has_a_test_pair():
    assert {n for n, _ in PAIRS} == {p.name for p in CONFIG_DIR.glob("*.yaml")}


def test_unknown_keys_rejected(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("cache_path: x.json\nbogus_key: 1\n")
    with pytest.raises(ValueError, match="unknown config keys"):
        load_config(p, C.PubchemConfig)


def test_paths_and_tuples_coerced():
    cfg = load_config(CONFIG_DIR / "isotopes.yaml", C.IsotopesConfig)
    assert isinstance(cfg.gt_path, Path)
    assert isinstance(cfg.models, tuple)
    assert isinstance(cfg.reps, tuple)
