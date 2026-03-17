# CLAUDE.md

## Project overview

shell-rft is a reinforcement fine-tuning project for Fireworks single-turn RFT. A model receives a natural-language task plus a toy filesystem description and must return exactly one shell command. A Python evaluator materializes the workspace, runs the command in a sandbox, and returns a binary reward based on correctness.

This is **not** a shell agent. It is a focused experiment for learning how Fireworks RFT behaves on a code-like task.

## Tech stack

- **Python** (managed with `uv`)
- **fireworks-ai[reward-kit]** for the evaluator
- **pytest** for testing
- **eval-protocol** CLI for Fireworks dataset/evaluator registration and RFT job creation
- **Base model**: `accounts/fireworks/models/qwen3-8b`

## Project layout

```
shell-rft/
├── pyproject.toml
├── uv.lock
├── data/                       # Generated JSONL datasets
│   ├── train.jsonl
│   ├── val.jsonl
│   └── test.jsonl
├── shell_rft/                  # Core library code
│   ├── generation/             # Task family generators
│   ├── prompts.py              # Prompt templates
│   └── schemas.py              # Data schemas
├── evaluators/
│   └── shell_rft_v0/           # Self-contained evaluator package (Fireworks Build SDK deploys this dir)
│       ├── main.py             # @reward_function entry point
│       ├── sandbox.py          # Workspace materialization + sandboxed execution
│       ├── policy.py           # Command allowlist/blocklist validation
│       └── normalize.py        # Stdout normalization
├── scripts/
│   ├── generate_dataset.py
│   └── run_local_baseline.py
└── tests/
```

The evaluator directory (`evaluators/shell_rft_v0/`) must be self-contained because Fireworks Build SDK packages it as a unit.

## Commands

```bash
# Setup
uv sync

# Run tests
uv run pytest

# Run evaluator tests specifically
cd evaluators/shell_rft_v0 && uv run pytest -vs

# Generate dataset
uv run python scripts/generate_dataset.py

# Run prompt-only baseline
uv run python scripts/run_local_baseline.py
```

## Key design constraints

### Dataset format
Each JSONL row has:
- `messages`: system + user messages (drives rollout generation)
- `workspace_spec`: files to materialize for evaluation
- `expected_stdout`: ground truth output
- `task_type`: one of `file_counting`, `content_search`, `topk_by_size`, `csv_filtering`

### Prompt contract
- System prompt: "You solve toy shell tasks. Return exactly one shell command. Do not include markdown, comments, or explanation. Assume the current working directory is the workspace root."
- User prompt: Task description + filesystem summary. Keep it short.

### Reward
Binary only for v0:
- `1.0` if: single command extracted, passes policy, executes successfully, normalized stdout matches expected
- `0.0` otherwise

Do not add partial credit unless there is clear evidence it is needed.

### Sandbox policy
**Allowed commands**: `find`, `grep`, `sort`, `head`, `tail`, `wc`, `cut`, `awk`, `sed`, `cat`, `xargs`, `tr`

**Disallowed**: `rm`, `mv`, `cp`, network tools, process control, package managers, shell control operators that create multiple commands, paths outside the temp workspace.

Each evaluation gets a fresh temp directory, a ~2s timeout, no network, and constrained CPU/memory.

### Task families (v0 — exactly four)
1. **File counting** — count files matching patterns; output is a single integer
2. **Content search** — find files/lines containing a string
3. **Top-k by size** — largest/smallest files by size
4. **CSV filtering** — filter/count rows by column values

Do not add new task families without good reason.

## Planning

- Use a single `.ai/plans/` directory for milestone plans. Do not separate specs from plans — each milestone gets one document that covers both design and implementation steps.
- The design doc at `.ai/OVERALL_DESIGN.md` is the top-level source of truth. Milestone plans expand on it with implementation detail.

## Style guidelines

- Keep things simple. This project values evaluator trustworthiness over cleverness.
- Avoid over-engineering — this is a focused experiment, not a production system.
- The evaluator must be conservative: false positives are worse than false negatives.
- When in doubt, refer to the design doc at `.ai/OVERALL_DESIGN.md`.

## Non-goals

Do not build toward: a production shell agent, multi-turn tool use, remote-agent training, arbitrary command execution, open-ended bash scripting, or benchmarking against frontier models.
