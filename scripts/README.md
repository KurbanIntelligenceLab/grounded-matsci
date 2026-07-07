# Scripts

There are no shell/Python wrappers here: every entry point is a CLI subcommand,

```bash
uv run grounded-matsci <subcommand> --config configs/<name>.yaml
```

(see `src/grounded_matsci/cli.py`). Nothing in this directory may contain scientific
logic; all experiment entry points live in `cli.py` and `workflows/`.
