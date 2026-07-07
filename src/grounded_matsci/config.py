"""Typed experiment configuration.

Every runtime experiment parameter (model lists, system prompts, worker counts, paths)
lives in versioned YAML under `configs/` and is validated here into a frozen dataclass
(rule D). Frozen scientific constants (tolerances, thresholds, reference tables) stay in
`domain/`; they are part of the registered method, not knobs.
"""

from __future__ import annotations

import dataclasses
import types
import typing
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


def _coerce(hint: Any, value: Any) -> Any:  # noqa: ANN401 - boundary coercion is dynamic
    """Coerce a YAML scalar/list into the annotated field type (Path, tuple, optional)."""
    origin = typing.get_origin(hint)
    if origin in (types.UnionType, typing.Union):
        members = [m for m in typing.get_args(hint) if m is not type(None)]
        if value is None:
            return None
        return _coerce(members[0], value) if members else value
    if hint is Path:
        return Path(value)
    if origin is tuple:
        return tuple(value)
    return value


def load_config[T](path: str | Path, cls: type[T]) -> T:
    """Load a YAML mapping and validate it into the frozen dataclass `cls`.

    Unknown keys are rejected; missing required keys raise from the constructor.
    """
    if not dataclasses.is_dataclass(cls):
        raise TypeError(f"{cls!r} is not a dataclass")
    raw = yaml.safe_load(Path(path).read_text())
    if not isinstance(raw, dict):
        raise TypeError(f"config {path} must be a YAML mapping")
    hints = typing.get_type_hints(cls)
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"unknown config keys in {path}: {sorted(unknown)}")
    kwargs = {k: _coerce(hints[k], v) for k, v in raw.items()}
    return typing.cast("T", cls(**kwargs))


@dataclass(frozen=True, slots=True)
class HarnessConfig:
    seed: int
    models: tuple[str, ...]
    system: str
    corpus_path: Path
    ckpt_path: Path
    rep: int = 0
    workers: int = 16
    log_every: int = 200
    limit: int | None = None
    concurrent: bool = True


@dataclass(frozen=True, slots=True)
class ArmsConfig:
    seed: int
    models: tuple[str, ...]
    system: str
    corpus_path: Path
    ckpt_path: Path
    conditions: tuple[str, ...]
    facts_path: Path
    named_sg_path: Path
    ef_lookup_path: Path
    rep: int = 0
    workers: int = 12
    log_every: int = 100
    limit: int | None = None


@dataclass(frozen=True, slots=True)
class SelfcheckConfig:
    seed: int
    models: tuple[str, ...]
    system: str
    prompts_path: Path
    ckpt_path: Path
    n: int = 8
    rep: int = 0
    workers: int = 12
    log_every: int = 100


@dataclass(frozen=True, slots=True)
class InlineConfConfig:
    seed: int
    models: tuple[str, ...]
    system: str
    conf_suffix: str
    prompts_path: Path
    ckpt_path: Path
    rep: int = 0
    workers: int = 16
    log_every: int = 100


@dataclass(frozen=True, slots=True)
class EndtaskConfig:
    seed: int
    models: tuple[str, ...]
    system: str
    tasks_path: Path
    ckpt_path: Path
    named_sg_path: Path
    ef_lookup_path: Path
    conditions: tuple[str, ...] = ("baseline", "mode_a", "mode_b")
    rep: int = 0
    workers: int = 12
    log_every: int = 50


@dataclass(frozen=True, slots=True)
class EndtaskDerivedConfig:
    seed: int
    models: tuple[str, ...]
    system: str
    tasks_path: Path
    ckpt_path: Path
    rep: int = 0
    workers: int = 16
    tol: float = 0.01
    log_every: int = 50


@dataclass(frozen=True, slots=True)
class CodataConfig:
    seed: int
    models: tuple[str, ...]
    system: str
    prompts_path: Path
    ckpt_path: Path
    rep: int = 0
    workers: int = 16
    rel_tol: float = 0.01
    log_every: int = 50


@dataclass(frozen=True, slots=True)
class Exp1Config:
    seed: int
    models: tuple[str, ...]
    system: str
    corpus_path: Path
    ef_lookup_path: Path
    selfcheck_detection_path: Path
    out_path: Path
    reps: tuple[int, ...] = (1, 2, 3)
    workers: int = 12


@dataclass(frozen=True, slots=True)
class Exp2Config:
    seed: int
    models: tuple[str, ...]
    system: str
    prompts_path: Path
    gt_path: Path
    out_path: Path
    reps: tuple[int, ...] = (1, 2, 3)
    workers: int = 12


@dataclass(frozen=True, slots=True)
class Exp3Config:
    seed: int
    models: tuple[str, ...]
    system: str
    corpus_path: Path
    named_sg_path: Path
    ef_lookup_path: Path
    out_path: Path
    reps: tuple[int, ...] = (1, 2, 3)
    workers: int = 12


@dataclass(frozen=True, slots=True)
class Exp4Config:
    seed: int
    models: tuple[str, ...]
    system: str
    gt_path: Path
    prompts_path: Path
    out_path: Path
    reps: tuple[int, ...] = (1, 2, 3)
    workers: int = 12


@dataclass(frozen=True, slots=True)
class PubchemConfig:
    cache_path: Path
