"""JSONL checkpoint-resume helpers.

Every runner appends one JSON row per completed cell to a checkpoint file and, on
restart, skips cells already present. The resume-side readers were duplicated across
seven runners; they are deduplicated here. The append side stays in each runner (each
has its own row schema and lock)."""

from __future__ import annotations

import contextlib
import json
from pathlib import Path
from typing import Any


def load_done_cells(ckpt_path: str | Path) -> set[str]:
    """Cell ids already present in a checkpoint JSONL (malformed lines skipped)."""
    done: set[str] = set()
    p = Path(ckpt_path)
    if p.exists():
        with p.open() as f:
            for ln in f:
                with contextlib.suppress(Exception):
                    done.add(json.loads(ln)["cell"])
    return done


def load_done_records(ckpt_path: str | Path) -> dict[str, dict[str, Any]]:
    """Cell id -> full record for a checkpoint JSONL (malformed lines skipped)."""
    done: dict[str, dict[str, Any]] = {}
    p = Path(ckpt_path)
    if p.exists():
        with p.open() as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    done[rec["cell"]] = rec
                except Exception:
                    pass
    return done
