"""Command-line entrypoints: `uv run grounded-matsci <subcommand> --config configs/<name>.yaml`.

Each subcommand loads its YAML config into a frozen dataclass, configures logging, seeds
the RNGs, snapshots the config + provenance manifest into `outputs/<experiment_id>/`, and
calls the corresponding workflow function. This module is plumbing only — the scientific
orchestration lives in `workflows/`.

The original bundle shipped `main()` entrypoints only for exp1..exp4; the remaining
runners were driven from an interactive kernel not included in the bundle. For those, the
ground-truth-kwargs wiring below reconstructs the in-bundle pattern (exp3's `gk_for`).

Most subcommands read frozen input files (corpus, ground truth, prompts) that are NOT in
this repository — see `data/manifests/handoff_missing.yaml`. Running them without that
data release fails with a FileNotFoundError.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from grounded_matsci import config as cfgmod
from grounded_matsci.config import load_config
from grounded_matsci.logging_config import configure_logging
from grounded_matsci.reproducibility import seed, start_run


def _load_json(path: Path) -> Any:  # noqa: ANN401 - raw JSON boundary
    return json.loads(Path(path).read_text())


def _make_gt_kwargs_for(
    named_sg_path: Path, ef_lookup_path: Path
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """The in-loop verify kwargs factory, per the exp3 runner's `gk_for` pattern."""
    named_sg = _load_json(named_sg_path)
    ef_lookup = _load_json(ef_lookup_path)
    sg_lookup = named_sg.get("token_accepted_sg", named_sg)

    def gk_for(rec: dict[str, Any]) -> dict[str, Any]:
        return {
            "enable_physics": True,
            "ef_lookup": ef_lookup,
            "named_sg_lookup": sg_lookup,
            "extra_known_names": [rec["subject"].lower()],
        }

    return gk_for


def _begin(args: argparse.Namespace, cls: type[Any]) -> Any:  # noqa: ANN401 - heterogeneous configs
    cfg = load_config(args.config, cls)
    configure_logging(json_logs=not args.debug)
    seed(cfg.seed)
    start_run(args.config, cfg.seed, debug=args.debug)
    return cfg


def _cmd_harness(args: argparse.Namespace) -> int:
    from grounded_matsci.workflows import harness

    cfg = _begin(args, cfgmod.HarnessConfig)
    corpus = _load_json(cfg.corpus_path)
    run = harness.run_baseline_concurrent if cfg.concurrent else harness.run_baseline_pass
    kwargs: dict[str, Any] = {"rep": cfg.rep, "limit": cfg.limit, "log_every": cfg.log_every}
    if cfg.concurrent:
        kwargs["workers"] = cfg.workers
    run(corpus, cfg.ckpt_path, list(cfg.models), cfg.system, **kwargs)
    return 0


def _cmd_arms(args: argparse.Namespace) -> int:
    from grounded_matsci.workflows import arms

    cfg = _begin(args, cfgmod.ArmsConfig)
    corpus = _load_json(cfg.corpus_path)
    fact_lookup = arms.build_fact_lookup(_load_json(cfg.facts_path))
    gk_for = _make_gt_kwargs_for(cfg.named_sg_path, cfg.ef_lookup_path)
    arms.run_arms(
        corpus,
        cfg.ckpt_path,
        list(cfg.conditions),
        gk_for,
        fact_lookup,
        list(cfg.models),
        cfg.system,
        rep=cfg.rep,
        workers=cfg.workers,
        log_every=cfg.log_every,
        limit=cfg.limit,
    )
    return 0


def _cmd_selfcheck(args: argparse.Namespace) -> int:
    from grounded_matsci.workflows import selfcheck

    cfg = _begin(args, cfgmod.SelfcheckConfig)
    prompts = _load_json(cfg.prompts_path)
    selfcheck.run_selfcheck(
        prompts,
        cfg.ckpt_path,
        lambda p: {},
        list(cfg.models),
        cfg.system,
        n=cfg.n,
        rep=cfg.rep,
        workers=cfg.workers,
        log_every=cfg.log_every,
    )
    return 0


def _cmd_inline_conf(args: argparse.Namespace) -> int:
    from grounded_matsci.workflows import inline_conf

    cfg = _begin(args, cfgmod.InlineConfConfig)
    prompts = _load_json(cfg.prompts_path)
    inline_conf.run_inline_conf(
        prompts,
        cfg.ckpt_path,
        list(cfg.models),
        cfg.system,
        cfg.conf_suffix,
        rep=cfg.rep,
        workers=cfg.workers,
        log_every=cfg.log_every,
    )
    return 0


def _cmd_endtask(args: argparse.Namespace) -> int:
    from grounded_matsci.workflows import endtask

    cfg = _begin(args, cfgmod.EndtaskConfig)
    tasks = _load_json(cfg.tasks_path)
    gk_for = _make_gt_kwargs_for(cfg.named_sg_path, cfg.ef_lookup_path)
    endtask.run_endtask(
        tasks,
        cfg.ckpt_path,
        gk_for,
        list(cfg.models),
        cfg.system,
        conditions=tuple(cfg.conditions),
        rep=cfg.rep,
        workers=cfg.workers,
        log_every=cfg.log_every,
    )
    return 0


def _cmd_endtask_derived(args: argparse.Namespace) -> int:
    from grounded_matsci.workflows import endtask_derived

    cfg = _begin(args, cfgmod.EndtaskDerivedConfig)
    tasks = _load_json(cfg.tasks_path)
    endtask_derived.run_endtask_derived(
        tasks,
        cfg.ckpt_path,
        list(cfg.models),
        cfg.system,
        rep=cfg.rep,
        workers=cfg.workers,
        tol=cfg.tol,
        log_every=cfg.log_every,
    )
    return 0


def _cmd_codata(args: argparse.Namespace) -> int:
    from grounded_matsci.workflows import codata

    cfg = _begin(args, cfgmod.CodataConfig)
    prompts = _load_json(cfg.prompts_path)
    codata.run_codata(
        prompts,
        cfg.ckpt_path,
        list(cfg.models),
        cfg.system,
        rep=cfg.rep,
        workers=cfg.workers,
        rel_tol=cfg.rel_tol,
        log_every=cfg.log_every,
    )
    return 0


def _cmd_exp1(args: argparse.Namespace) -> int:
    from grounded_matsci.workflows import exp1_two_stage

    cfg = _begin(args, cfgmod.Exp1Config)
    exp1_two_stage.run(
        cfg.corpus_path,
        cfg.ef_lookup_path,
        cfg.selfcheck_detection_path,
        cfg.out_path,
        list(cfg.models),
        cfg.system,
        reps=tuple(cfg.reps),
        workers=cfg.workers,
    )
    return 0


def _cmd_exp2(args: argparse.Namespace) -> int:
    from grounded_matsci.workflows import exp2_gated_codata

    cfg = _begin(args, cfgmod.Exp2Config)
    exp2_gated_codata.run(
        cfg.prompts_path,
        cfg.gt_path,
        cfg.out_path,
        list(cfg.models),
        cfg.system,
        reps=tuple(cfg.reps),
        workers=cfg.workers,
    )
    return 0


def _cmd_exp3(args: argparse.Namespace) -> int:
    from grounded_matsci.workflows import exp3_small_models

    cfg = _begin(args, cfgmod.Exp3Config)
    exp3_small_models.run(
        cfg.corpus_path,
        cfg.named_sg_path,
        cfg.ef_lookup_path,
        cfg.out_path,
        list(cfg.models),
        cfg.system,
        reps=tuple(cfg.reps),
        workers=cfg.workers,
    )
    return 0


def _cmd_exp4(args: argparse.Namespace) -> int:
    from grounded_matsci.workflows import exp4_isotopes

    cfg = _begin(args, cfgmod.Exp4Config)
    exp4_isotopes.run(
        cfg.gt_path,
        cfg.prompts_path,
        cfg.out_path,
        list(cfg.models),
        cfg.system,
        reps=tuple(cfg.reps),
        workers=cfg.workers,
    )
    return 0


_COMMANDS: dict[str, tuple[Callable[[argparse.Namespace], int], str]] = {
    "harness": (_cmd_harness, "baseline (condition 1) holdout generation pass"),
    "arms": (_cmd_arms, "intervention arms (conditions 2-5) over the frozen corpus"),
    "selfcheck": (_cmd_selfcheck, "sampling-consistency detection (SelfCheckGPT-style)"),
    "inline-conf": (_cmd_inline_conf, "inline verbalized-confidence arm"),
    "endtask": (_cmd_endtask, "end-task propagation runner (item 1)"),
    "endtask-derived": (_cmd_endtask_derived, "derived-quantity end-task tier (item 3)"),
    "codata": (_cmd_codata, "physical-constants transfer domain (item 3)"),
    "exp1": (_cmd_exp1, "EXP1 two-stage triage detector"),
    "exp2": (_cmd_exp2, "EXP2 gated CODATA rerun"),
    "exp3": (_cmd_exp3, "EXP3 small-model capability probe"),
    "exp4": (_cmd_exp4, "EXP4 isotope half-life domain"),
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="grounded-matsci", description="Grounded verification experiment runners"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for name, (_fn, help_text) in _COMMANDS.items():
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--config", type=Path, required=True, help="YAML config under configs/")
        p.add_argument(
            "--debug", action="store_true", help="allow a dirty git tree (non-final run)"
        )
    args = parser.parse_args(argv)
    fn, _ = _COMMANDS[args.command]
    return fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
