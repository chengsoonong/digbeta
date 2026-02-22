# Putative Residues

Visualise feature importance in a protein sequence by colouring the corresponding 3D structure.

## Setup

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

## Run

```bash
uv run marimo edit --watch notebook.py
```

## Using Claude with Marimo

There are two ways to use Claude to develop marimo notebooks.

### Claude Code (CLI)

Run `claude` in the terminal alongside `marimo edit`. Claude Code operates on
`notebook.py` as a regular Python file — it can see the entire codebase, make
multi-cell changes, manage dependencies in `pyproject.toml`, and run shell
commands. Best for larger refactors and tasks that span multiple cells or files.

### Marimo's Built-in AI (`Ctrl+E`)

Marimo has a native AI assistant in the editor. Press `Ctrl+E` on any cell to
generate or edit its contents inline. This keeps you in the notebook UI with a
tight feedback loop, but operates one cell at a time.

Requires `ANTHROPIC_API_KEY` in your environment and configuration in
`~/.marimo.toml` (easiest to set via Settings > AI tab in the editor):

```toml
[ai.open_ai]
model = "anthropic/claude-sonnet-4-5-20250929"

[ai.anthropic]
api_key = "sk-ant-..."
```
