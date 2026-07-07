# Scripts

There are no shell/Python wrappers here: every entry point is a CLI subcommand,

```bash
uv run grounded-matsci <subcommand> --config configs/<name>.yaml
```

(see `src/grounded_matsci/cli.py`). Per CODING_RULES rule A, nothing in this directory
may contain scientific logic.
