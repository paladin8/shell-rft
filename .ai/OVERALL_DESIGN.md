# shell-rft Design Doc

## Summary

`shell-rft` is a small reinforcement fine-tuning project built specifically for the Fireworks single-turn RFT workflow. The model sees a natural-language task plus a compact description of a toy filesystem and must return exactly one shell command. A Python evaluator materializes the toy workspace, executes the command in a sandbox, and returns a reward in `[0, 1]` based on correctness.

This is not a shell agent. It ([docs.fireworks.ai](https://docs.fireworks.ai/fine-tuning/reinforcement-fine-tuning-models))nd for learning how Fireworks RFT behaves on a code-like task with multiple valid solutions.

## Why this project

This is a good fit for Fireworks RFT because:

* the task is single-turn,
* the reward is programmatic and fast,
* correctness can be checked by execution,
* many commands can be valid for the same task,
* the environment can be synthesized at scale.

That means we should use Fireworks’ **single-turn training** path with a **local evaluator**, not remote agents. Remote environments are for multi-turn agents, private infrastructure, or existing services; this project needs none of that.

## Decision summary

### Training path

Use **Fireworks single-turn RFT** with:

* a JSONL dataset of prompts,
* a local Python evaluator built with `fireworks-ai[reward-kit]`,
* the `eval-protocol` CLI to register the dataset and evaluator and create the RFT job.

### Base model

Start with **`accounts/fireworks/models/qwen3-8b`**.

Reasoning:

* it is meaningfully stronger than the tutorial-scale models,
* it is still small enough for fast iteration,
* it is under Fireworks’ free-tuning threshold of 16B,
* Fireworks currently recommends Qwen 3 8B as one of its small, fast general-purpose models and also points to Qwen 3 8B as a low-latency alternative for coding/agentic use cases.

We are not starting with `qwen3-0p6b`. Fireworks uses that in tutorials because it is tiny and cheap, not because it is the right model for a code-like shell task.

### Initial reward design

Use a **mostly binary execution-based reward**.

For v0:

* `0.0` if the output is not a single valid command
* `0.0` if the command violates policy
* `0.0` if execution fails or output is wrong
* `1.0` if normalized stdout exactly matches the expected stdout

This is intentionally blunt. Fireworks explicitly recommends starting with the simplest evaluator that captures the core requirement, and binary scoring works well for many tasks.

### Task scope

Keep v0 narrow:

* file counting,
* content search,
* top-k by file size,
* simple CSV filtering.

No multi-turn tasks. No planning. No shell scripts. No network. No arbitrary system administration.

## End-to-end Fireworks workflow

The project will follow this concrete path:

1. Generate synthetic JSONL training and eval data locally.
2. Implement a local evaluator in its own directory using `fireworks-ai[reward-kit]`.
3. Test the evaluator locally with Python and with `pytest`.
4. Register the dataset and evaluator with Fireworks.
5. Launch a single-turn RFT job with `eval-protocol create rft` using `accounts/fireworks/models/qwen3-8b`.
6. Monitor reward and held-out accuracy.
7. Compare against a prompt-only baseline.
8. Deploy the tuned model and optionally use grammar-constrained inference at serving time to force one-command output.

## What Fireworks expects from us

### Dataset shape

Fireworks RFT expects a JSONL dataset where each example contains `messages`. The docs also show that examples can include additional ground-truth fields for the evaluator.

Our dataset rows will look like this:

```json
{
  "messages": [
    {
      "role": "system",
      "content": "Return exactly one shell command. No markdown. No explanation. Assume the current working directory is the workspace root."
    },
    {
      "role": "user",
      "content": "Task: Find the three largest .log files under app/, excluding app/archive/. Print only the relative paths, one per line, largest first.

Filesystem summary:
- app/logs/a.log (1200 bytes)
- app/logs/b.log (5000 bytes)
- app/archive/old.log (9000 bytes)
- app/services/auth/debug.log (3000 bytes)"
    }
  ],
  "workspace_spec": {
    "files": [
      {"path": "app/logs/a.log", "content": "..."},
      {"path": "app/logs/b.log", "content": "..."},
      {"path": "app/archive/old.log", "content": "..."},
      {"path": "app/services/auth/debug.log", "content": "..."}
    ]
  },
  "expected_stdout": "app/logs/b.log
app/services/auth/debug.log
app/logs/a.log
",
  "task_type": "topk_by_size"
}
```

The important part is that `messages` drives rollout generation, while `workspace_spec`, `expected_stdout`, and `task_type` are consumed by the evaluator.

### Evaluator shape

Fireworks’ Build SDK packages evaluator code from the directory containing the reward function, so the evaluator must live in its own directory.

Proposed layout:

```text
shell-rft/
├── pyproject.toml
├── uv.lock
├── data/
│   ├── train.jsonl
│   ├── val.jsonl
│   └── test.jsonl
├── shell_rft/
│   ├── generation/
│   ├── prompts.py
│   └── schemas.py
├── evaluators/
│   └── shell_rft_v0/
│       ├── main.py
│       ├── sandbox.py
│       ├── policy.py
│       └── normalize.py
├── scripts/
│   ├── generate_dataset.py
│   └── run_local_baseline.py
└── tests/
```

Evaluator sketch:

```python
from fireworks import reward_function

@reward_function(id="shell-rft-v0")
def evaluate(messages, workspace_spec=None, expected_stdout=None, task_type=None, **kwargs):
    command = extract_single_command(messages[-1]["content"])
    if command is None:
        return {"score": 0.0}

    violation = validate_command(command)
    if violation is not None:
        return {"score": 0.0, "reason": violation}

    with materialize_workspace(workspace_spec) as workdir:
        result = run_command(command, cwd=workdir, timeout_s=2)

    actual = normalize_stdout(result.stdout, task_type=task_type)
    expected = normalize_stdout(expected_stdout, task_type=task_type)

    if result.exit_code == 0 and actual == expected:
        return {"score": 1.0}
    return {"score": 0.0}
```

One subtle point: the evaluator scores the model’s output message, not our prompt. So the rollout output must be a single command with no extra chatter.

## Model choice

### Chosen model

`accounts/fireworks/models/qwen3-8b`

### Why this model

This project needs a model that is:

* small enough for cheap/frequent iteration,
* good enough at shell-like code patterns to show measurable learning,
* available in Fireworks’ current fine-tuning stack.

Qwen3-8B is the right compromise. A 0.6B model is useful for tutorials, but it is likely too weak to make the experiment interesting. A much larger model would make the toy project slower and less useful for iteration.

### One caution

Fireworks documents that Qwen3 models default to reasoning-on behavior at inference time. For deployment and local baseline evaluation, we should disable reasoning when possible so the model is less likely to emit extra text before the command. The training prompt should also be strict: one command, no markdown, no explanation.

## Prompt contract

System prompt:

```text
You solve toy shell tasks.
Return exactly one shell command.
Do not include markdown, comments, or explanation.
Assume the current working directory is the workspace root.
```

User prompt template:

```text
Task:
{task}

Filesystem summary:
{filesystem_summary}
```

This stays deliberately short. The project is about reward learning, not giant prompt engineering.

## Task families for v0

We will implement exactly four task families.

### 1. File counting

Examples:

* Count `.txt` files under `docs/`.
* Count how many files contain an exact phrase.

Expected output is usually a single integer.

### 2. Content search

Examples:

* Print the relative paths of files containing `timeout exceeded`.
* Print matching lines from files under `logs/`.

### 3. Top-k by size

Examples:

* Print the three largest `.log` files under `app/`, excluding `archive/`.

### 4. Simple CSV filtering

Examples:

* Print usernames whose `usage_pct` is greater than 80.
* Count rows where `status` is `failed`.

These are enough to exercise `find`, `grep`, `sort`, `head`, `wc`, and simple text processing without turning the task family into a mess.

## Safety and sandbox policy

The evaluator runs model-generated shell, so the command space must be tightly restricted.

### Allowed tools for v0

* `find`
* `grep`
* `sort`
* `head`
* `tail`
* `wc`
* `cut`
* `awk`
* `sed`
* `cat`
* `xargs`
* `tr`

### Disallowed

* `rm`, `mv`, `cp`
* network tools
* process-control tools
* package managers
* shell control operators that create multiple commands
* anything touching paths outside the temp workspace

### Sandbox requirements

* fresh temp directory per eval,
* timeout around 2 seconds,
* no network,
* constrained CPU and memory,
* execution only inside the generated workspace.

This policy is conservative on purpose. The first experiment should optimize for evaluator trustworthiness, not realism.

## Reward design

For the first real run, keep the reward binary.

```text
score = 1.0 if:
- exactly one command is extracted
- command passes policy validation
- command executes successfully
- normalized stdout exactly matches expected_stdout

otherwise score = 0.0
```

Why binary first:

* Fireworks explicitly recommends starting simple,
* exact-match executable tasks are where binary reward works well,
* partial-credit schemes create extra reward-hacking surface area,
* we can add shaping later if we see clear evidence it is needed.

If we later relax this, the first addition should be a small partial reward for correct rows with wrong ordering on tasks where that distinction is meaningful.

## Initial Fireworks training setup

### Local setup

```bash
uv init
uv add fireworks-ai[reward-kit] pytest
export FIREWORKS_API_KEY=<your_key>
```

### Dataset registration

```bash
eval-protocol create dataset shell-rft-train --file data/train.jsonl
eval-protocol create dataset shell-rft-val --file data/val.jsonl
```

### Evaluator registration

The Fireworks docs use `pytest` as the path that both validates and registers the evaluator. We will keep that pattern.

```bash
cd evaluators/shell_rft_v0
pytest -vs
```

### RFT job creation

We will create the first job from the CLI, not the UI, because it is easier to reproduce and version.

```bash
eval-protocol create rft \
  --base-model accounts/fireworks/models/qwen3-8b
```

The exact command flags may evolve as we fill in dataset and evaluator identifiers, but the project should be built around this `eval-protocol` path.

## Rollout parameters for the first run

Fireworks recommends starting with defaults, and their docs say many successful RFT jobs do exactly that. We should mostly follow that advice.

Starting plan:

* keep default training parameters,
* keep `n=4` rollouts per prompt,
* keep standard sampling defaults unless local testing shows obvious problems,
* reduce max response length only if needed after the first smoke run.

For this task, I expect we will end up with a relatively small max token budget because outputs should be one shell command, but we should not optimize that before we have one baseline run working.

## Baselines

Before RFT, measure two baselines.

### Baseline A: prompt-only

Evaluate the raw base model on the held-out set with the exact same prompt contract and evaluator.

### Baseline B: prompt-only plus constrained output at inference

For deployment or local inference experiments, use Fireworks grammar mode to constrain output to a single command-shaped string. This is not a substitute for RFT, but it will tell us how much of the problem is just formatting.

The important comparison is:

* prompt-only,
* prompt-only plus output constraint,
* tuned model.

If prompt-only already solves nearly everything, the task is too easy and the project needs harder examples, not fancier RL.

## Metrics

Track a small set of metrics only.

* exact-match rate,
* mean reward,
* parse-success rate,
* policy-violation rate,
* execution-success rate,
* per-task-family exact-match rate.

Everything else is secondary.

## Success criteria

The project is successful if:

1. the evaluator is trustworthy on manual inspection,
2. the end-to-end Fireworks workflow runs cleanly,
3. the tuned model beats the prompt-only baseline on held-out exact-match rate,
4. the improvement is not just fewer formatting failures.

## Milestones

### Milestone 1

Repo skeleton, schema, prompt contract, and one task family.

### Milestone 2

Local evaluator with sandbox and tests.

### Milestone 3

All four task families generating train/val/test JSONL.

### Milestone 4

Prompt-only baseline on held-out data.

### Milestone 5

First Fireworks RFT run on `accounts/fireworks/models/qwen3-8b`.

### Milestone 6

Short analysis memo: where the gains came from, where the model still fails, and whether to expand scope.

## Non-goals

* a production shell agent,
* multi-turn tool use,
* remote-agent Fireworks training,
* arbitrary command execution,
* open-ended bash scripting,
* benchmarking against large frontier coding models.

## v0 recommendation

Keep this disciplined.

Use Fireworks single-turn RFT, `accounts/fireworks/models/qwen3-8b`, a local execution-based evaluator, a binary reward, and a narrow synthetic dataset. That is the cleanest way to make the project teach us something real about reward design and model improvement rather than just creating an overengineered toy.
