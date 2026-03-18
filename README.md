# shell-rft

A reinforcement fine-tuning experiment using [Fireworks single-turn RFT](https://docs.fireworks.ai/fine-tuning/reinforcement-fine-tuning-models). A model receives a natural-language task plus a toy filesystem description and must return exactly one shell command. A Python evaluator materializes the workspace, runs the command in a sandbox, and scores correctness with a binary reward.

This is not a shell agent. It's a focused experiment for learning how Fireworks RFT behaves on a code-like task with multiple valid solutions.

## How it works

1. **Synthetic data generation** — task generators produce JSONL rows, each with a prompt, a workspace spec (files to create), and expected stdout.
2. **Evaluation** — the evaluator extracts a single command from the model's response, validates it against a safety policy, executes it in a sandboxed temp directory, and compares normalized stdout to the expected output.
3. **Training** — Fireworks RFT uses the binary reward signal to fine-tune `qwen3-8b`.

## Task families (v0)


| Family         | Example task                             | Example output                  |
| -------------- | ---------------------------------------- | ------------------------------- |
| File counting  | Count `.py` files under `src/`           | `3`                             |
| Content search | Find files containing `timeout exceeded` | `logs/app.log`                  |
| Top-k by size  | Three largest `.log` files               | `big.log` `med.log` `small.log` |
| CSV filtering  | Usernames with `usage_pct > 80`          | `alice` `bob`                   |


## Quick start

```bash
# Install dependencies
uv sync

# Run all tests
uv run pytest

# Run evaluator tests
cd evaluators/shell_rft_v0 && uv run pytest -vs

# Generate dataset
uv run python scripts/generate_dataset.py

# Run prompt-only baseline
uv run python scripts/run_local_baseline.py
```

## Project layout

```
shell-rft/
├── shell_rft/              # Core library (schemas, prompts, task generators)
│   ├── generation/          # Task family generators
│   ├── prompts.py           # Prompt templates
│   └── schemas.py           # Data schemas
├── evaluators/
│   └── shell_rft_v0/        # Self-contained evaluator (deployed by Fireworks Build SDK)
│       ├── main.py           # @reward_function entry point
│       ├── policy.py         # Command allowlist/blocklist
│       ├── sandbox.py        # Workspace materialization + execution
│       └── normalize.py      # Stdout normalization
├── scripts/                 # Dataset generation, baseline evaluation
├── data/                    # Generated JSONL datasets
└── tests/                   # Root-level tests
```

## License

MIT