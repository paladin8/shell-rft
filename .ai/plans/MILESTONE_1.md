# Milestone 1: Schema, Prompt Contract, and First Task Family

**Status: COMPLETE**

**Goal:** Implement the file-counting task family generator and wire up the dataset generation script to produce JSONL files.

**Architecture:** A `file_counting.py` module generates `Example` objects for four sub-types of counting tasks (by extension, by name pattern, all in subtree, files containing string). A generation registry in `__init__.py` maps task types to generator functions. The `generate_dataset.py` script uses the registry to produce train/val/test JSONL splits with per-split seeding.

**Tech Stack:** Python 3.11+, standard library only (random, json, argparse, dataclasses), pytest

---

## Status

**Partially complete.** The bootstrap delivered the repo skeleton, schema, and prompt contract. What remains is the first task family generator and dataset generation wiring.

### What's already done

- `shell_rft/schemas.py` — `FileSpec`, `WorkspaceSpec`, `Example`, `TaskType`
- `shell_rft/prompts.py` — `SYSTEM_PROMPT`, `USER_PROMPT_TEMPLATE`
- `shell_rft/generation/__init__.py` — empty
- `scripts/generate_dataset.py` — stub (`NotImplementedError`)
- `tests/test_smoke.py` — 3 passing smoke tests

---

## Design

### File-counting task family generator

**Module**: `shell_rft/generation/file_counting.py`

A function that produces `Example` objects for file-counting tasks. Each example:

- Creates a synthetic workspace with a randomized directory tree
- Poses a natural-language task asking the model to count files matching some criteria
- Provides the expected stdout (a single integer followed by a newline)

#### Task variations

The generator should produce a mix of these sub-types, controlled by a seeded RNG:

| Sub-type | Example task | Key commands |
|----------|-------------|--------------|
| Count by extension | "Count the `.log` files under `app/`" | `find ... \| wc -l` |
| Count by name pattern | "Count files whose name starts with `test_`" | `find ... -name 'test_*' \| wc -l` |
| Count all files in subtree | "How many files are in `src/`?" | `find ... -type f \| wc -l` |
| Count files containing string | "How many files under `logs/` contain `ERROR`?" | `grep -rl ... \| wc -l` |

These sub-types are internal to the generator — they do not appear in the dataset schema. The `task_type` field is always `"file_counting"`.

#### Workspace synthesis

Each workspace should:

- Have 5–20 files across 2–5 directories
- Use realistic-looking paths (e.g., `src/`, `logs/`, `data/`, `app/config/`)
- Include a mix of extensions (`.py`, `.txt`, `.log`, `.csv`, `.json`)
- Include file content that is short (1–5 lines) and plausible
- Be fully deterministic given a seed

#### Filesystem summary format

The filesystem summary in the user prompt lists each file path:

```
- src/main.py
- src/utils.py
- logs/app.log
- logs/debug.log
- data/users.csv
```

No file sizes or content shown (file counting doesn't need them).

#### Expected stdout

Always a single integer followed by `\n`. For example: `"3\n"`.

#### Function signature

```python
def generate_file_counting_examples(
    n: int,
    rng: random.Random,
) -> list[Example]:
    """Generate n file-counting examples using the given RNG."""
```

### Generation registry

**Module**: `shell_rft/generation/__init__.py`

Export a registry mapping task types to generator functions:

```python
GENERATORS: dict[str, Callable[[int, random.Random], list[Example]]]
```

For milestone 1 this contains only `"file_counting"`. Future milestones add entries.

### Dataset generation script

**Module**: `scripts/generate_dataset.py`

Wire up the stub to:

1. Accept CLI args: `--seed` (default 42), `--train-n` (default 200), `--val-n` (default 50), `--test-n` (default 50), `--output-dir` (default `data/`)
2. For each split, call the generator with the appropriate count and a per-split seed derived from the base seed
3. Write each split as a JSONL file where each line is a JSON object with keys: `messages`, `workspace_spec`, `expected_stdout`, `task_type`

Use `argparse` for CLI. Serialize with `dataclasses.asdict`.

### Design constraints

- **Determinism**: All randomness flows through `random.Random` instances seeded from CLI args. No global random state.
- **No unnecessary deps**: The generator uses only the standard library.
- **Prompt contract**: The generator must use `SYSTEM_PROMPT` and `USER_PROMPT_TEMPLATE` from `shell_rft.prompts` — not hardcode prompt strings.
- **Schema reuse**: The generator produces `Example` objects from `shell_rft.schemas`.
- **Correctness over variety**: The generator must compute the expected count from the workspace it built, not from a separate calculation.

### Out of scope

- Other three task families (milestone 3)
- Evaluator implementation (milestone 2)
- Fireworks registration (milestone 5)
- Partial credit or non-binary reward
- File content in the filesystem summary (not needed for counting tasks)

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `shell_rft/generation/file_counting.py` | Create | Generate `Example` objects for file-counting tasks across 4 sub-types |
| `shell_rft/generation/__init__.py` | Modify | Export `GENERATORS` registry mapping task type names to generator functions |
| `scripts/generate_dataset.py` | Modify | CLI tool producing train/val/test JSONL from registered generators |
| `tests/test_file_counting.py` | Create | Unit tests for the generator (count, determinism, structure, prompt contract, correctness) |
| `tests/test_generate_dataset.py` | Create | Integration tests for the script (file creation, JSONL validity) |

---

## Chunk 1: File-counting generator

### Task 1: Generator tests and implementation

**Files:**
- Create: `tests/test_file_counting.py`
- Create: `shell_rft/generation/file_counting.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_file_counting.py`:

```python
"""Tests for the file-counting task family generator."""

import random
import re

from shell_rft.generation.file_counting import generate_file_counting_examples
from shell_rft.prompts import SYSTEM_PROMPT


def test_generates_requested_count():
    examples = generate_file_counting_examples(10, random.Random(42))
    assert len(examples) == 10


def test_deterministic_with_seed():
    a = generate_file_counting_examples(5, random.Random(99))
    b = generate_file_counting_examples(5, random.Random(99))
    for x, y in zip(a, b):
        assert x.expected_stdout == y.expected_stdout
        assert x.messages == y.messages


def test_example_structure():
    for ex in generate_file_counting_examples(20, random.Random(42)):
        assert ex.task_type == "file_counting"
        assert len(ex.messages) == 2
        assert len(ex.workspace_spec.files) > 0
        assert re.match(r"^\d+\n$", ex.expected_stdout)


def test_messages_use_prompt_contract():
    for ex in generate_file_counting_examples(5, random.Random(42)):
        assert ex.messages[0] == {"role": "system", "content": SYSTEM_PROMPT}
        assert ex.messages[1]["role"] == "user"
        assert "Task:" in ex.messages[1]["content"]
        assert "Filesystem summary:" in ex.messages[1]["content"]


def test_workspace_files_exist_in_summary():
    for ex in generate_file_counting_examples(10, random.Random(42)):
        user_msg = ex.messages[1]["content"]
        for f in ex.workspace_spec.files:
            assert f.path in user_msg


def test_expected_stdout_is_non_negative():
    """All file counts must be non-negative integers."""
    for ex in generate_file_counting_examples(50, random.Random(42)):
        count = int(ex.expected_stdout.strip())
        assert count >= 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_file_counting.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'shell_rft.generation.file_counting'`

- [ ] **Step 3: Implement the generator**

Create `shell_rft/generation/file_counting.py`:

```python
"""File-counting task family generator."""

from __future__ import annotations

import random

from shell_rft.prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE
from shell_rft.schemas import Example, FileSpec, WorkspaceSpec

_DIRS = ["src", "logs", "data", "app", "lib", "tests", "docs", "config"]
_EXTENSIONS = [".py", ".txt", ".log", ".csv", ".json"]
_STEMS = [
    "main", "utils", "config", "app", "debug", "error",
    "output", "report", "index", "helper",
]
_CONTENT_LINES = [
    "# placeholder",
    "import os",
    "ERROR: connection refused",
    "DEBUG: starting service",
    "name,value",
    "hello world",
    "TODO: fix this",
    "print('hello')",
    "status=ok",
    "timeout exceeded",
]


def _build_workspace(rng: random.Random) -> list[FileSpec]:
    """Create a random set of files across a small directory tree."""
    n_dirs = rng.randint(2, 5)
    n_files = rng.randint(5, 20)
    dirs = rng.sample(_DIRS, n_dirs)

    files: list[FileSpec] = []
    seen: set[str] = set()

    for i in range(n_files):
        d = rng.choice(dirs)
        stem = rng.choice(_STEMS)
        ext = rng.choice(_EXTENSIONS)
        prefix = "test_" if rng.random() < 0.25 else ""
        path = f"{d}/{prefix}{stem}{ext}"

        if path in seen:
            path = f"{d}/{prefix}{stem}_{i}{ext}"
        if path in seen:
            continue
        seen.add(path)

        n_lines = rng.randint(1, 5)
        content = "\n".join(rng.choices(_CONTENT_LINES, k=n_lines)) + "\n"
        files.append(FileSpec(path=path, content=content))

    return files


def _filesystem_summary(files: list[FileSpec]) -> str:
    return "\n".join(f"- {f.path}" for f in files)


def _make_example(task: str, files: list[FileSpec], count: int) -> Example:
    user_content = USER_PROMPT_TEMPLATE.format(
        task=task,
        filesystem_summary=_filesystem_summary(files),
    )
    return Example(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        workspace_spec=WorkspaceSpec(files=files),
        expected_stdout=f"{count}\n",
        task_type="file_counting",
    )


def _count_by_extension(rng: random.Random) -> Example:
    files = _build_workspace(rng)
    ext = rng.choice(_EXTENSIONS)
    dirs = sorted({f.path.split("/")[0] for f in files})
    target_dir = rng.choice(dirs)
    count = sum(
        1 for f in files
        if f.path.startswith(f"{target_dir}/") and f.path.endswith(ext)
    )
    task = f"Count the {ext} files under {target_dir}/."
    return _make_example(task, files, count)


def _count_by_name_pattern(rng: random.Random) -> Example:
    files = _build_workspace(rng)
    count = sum(
        1 for f in files
        if f.path.split("/")[-1].startswith("test_")
    )
    task = 'Count files whose name starts with "test_".'
    return _make_example(task, files, count)


def _count_all_in_subtree(rng: random.Random) -> Example:
    files = _build_workspace(rng)
    dirs = sorted({f.path.split("/")[0] for f in files})
    target_dir = rng.choice(dirs)
    count = sum(1 for f in files if f.path.startswith(f"{target_dir}/"))
    task = f"How many files are in {target_dir}/?"
    return _make_example(task, files, count)


def _count_files_containing(rng: random.Random) -> Example:
    files = _build_workspace(rng)
    search_term = rng.choice(["ERROR", "TODO", "import", "hello", "status"])
    count = sum(1 for f in files if search_term in f.content)
    task = f'How many files contain "{search_term}"?'
    return _make_example(task, files, count)


_SUB_TYPES = [
    _count_by_extension,
    _count_by_name_pattern,
    _count_all_in_subtree,
    _count_files_containing,
]


def generate_file_counting_examples(
    n: int,
    rng: random.Random,
) -> list[Example]:
    """Generate n file-counting examples using the given RNG."""
    return [rng.choice(_SUB_TYPES)(rng) for _ in range(n)]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_file_counting.py -v`
Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add shell_rft/generation/file_counting.py tests/test_file_counting.py
git commit -m "feat: add file-counting task family generator with tests"
```

### Task 2: Generation registry

**Files:**
- Modify: `shell_rft/generation/__init__.py`
- Create: `tests/test_generation_registry.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_generation_registry.py`:

```python
"""Tests for the generation registry."""

from shell_rft.generation import GENERATORS


def test_registry_contains_file_counting():
    assert "file_counting" in GENERATORS


def test_registry_values_are_callable():
    for gen_fn in GENERATORS.values():
        assert callable(gen_fn)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_generation_registry.py -v`
Expected: FAIL with `ImportError: cannot import name 'GENERATORS'`

- [ ] **Step 3: Update the registry**

Replace contents of `shell_rft/generation/__init__.py` with:

```python
"""Task family generators."""

from shell_rft.generation.file_counting import generate_file_counting_examples

GENERATORS = {
    "file_counting": generate_file_counting_examples,
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_generation_registry.py -v`
Expected: All 2 tests PASS

- [ ] **Step 5: Commit**

```bash
git add shell_rft/generation/__init__.py tests/test_generation_registry.py
git commit -m "feat: add generation registry with file_counting entry"
```

---

## Chunk 2: Dataset generation script

### Task 3: Dataset generation script

**Files:**
- Modify: `scripts/generate_dataset.py`
- Create: `tests/test_generate_dataset.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_generate_dataset.py`:

```python
"""Integration tests for the dataset generation script."""

import json
import subprocess
import sys
from pathlib import Path


def test_script_produces_jsonl_files(tmp_path):
    result = subprocess.run(
        [
            sys.executable, "scripts/generate_dataset.py",
            "--train-n", "5",
            "--val-n", "3",
            "--test-n", "2",
            "--output-dir", str(tmp_path),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr

    for name, expected_n in [("train", 5), ("val", 3), ("test", 2)]:
        path = tmp_path / f"{name}.jsonl"
        assert path.exists(), f"{name}.jsonl not found"
        lines = path.read_text().strip().splitlines()
        assert len(lines) == expected_n, (
            f"{name}.jsonl has {len(lines)} lines, expected {expected_n}"
        )


def test_jsonl_rows_are_valid(tmp_path):
    subprocess.run(
        [
            sys.executable, "scripts/generate_dataset.py",
            "--train-n", "3",
            "--val-n", "1",
            "--test-n", "1",
            "--output-dir", str(tmp_path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    for jsonl_file in tmp_path.glob("*.jsonl"):
        for line in jsonl_file.read_text().strip().splitlines():
            row = json.loads(line)
            assert "messages" in row
            assert "workspace_spec" in row
            assert "expected_stdout" in row
            assert "task_type" in row
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_generate_dataset.py -v`
Expected: FAIL (NotImplementedError from the stub script)

- [ ] **Step 3: Implement the script**

Replace contents of `scripts/generate_dataset.py` with:

```python
"""Generate train/val/test JSONL datasets."""

import argparse
import json
import random
from dataclasses import asdict
from pathlib import Path

from shell_rft.generation import GENERATORS


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate shell-rft datasets")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-n", type=int, default=200)
    parser.add_argument("--val-n", type=int, default=50)
    parser.add_argument("--test-n", type=int, default=50)
    parser.add_argument("--output-dir", type=str, default="data")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    splits = [
        ("train", args.train_n, args.seed),
        ("val", args.val_n, args.seed + 1),
        ("test", args.test_n, args.seed + 2),
    ]

    for split_name, count, split_seed in splits:
        rng = random.Random(split_seed)
        examples = []
        generators = list(GENERATORS.values())
        per_gen = count // len(generators)
        remainder = count % len(generators)

        for i, gen_fn in enumerate(generators):
            n = per_gen + (1 if i < remainder else 0)
            examples.extend(gen_fn(n, rng))

        rng.shuffle(examples)

        path = output_dir / f"{split_name}.jsonl"
        with open(path, "w") as f:
            for ex in examples:
                f.write(json.dumps(asdict(ex)) + "\n")

        print(f"Wrote {len(examples)} examples to {path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_generate_dataset.py -v`
Expected: All 2 tests PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/generate_dataset.py tests/test_generate_dataset.py
git commit -m "feat: wire up dataset generation script with CLI"
```

### Task 4: Final verification

- [ ] **Step 1: Run all tests**

Run: `uv run pytest -v`
Expected: All 13 tests PASS (3 smoke + 6 generator + 2 registry + 2 script)

- [ ] **Step 2: Generate a small dataset and spot-check**

Run: `uv run python scripts/generate_dataset.py --train-n 5 --val-n 2 --test-n 2 --output-dir /tmp/shell-rft-check`

Inspect first line: `head -1 /tmp/shell-rft-check/train.jsonl | python -m json.tool`

Verify the output has all four keys and the structure looks correct.

- [ ] **Step 3: Run evaluator tests too**

Run: `cd evaluators/shell_rft_v0 && uv run pytest -vs`
Expected: 2 existing evaluator tests still PASS

---

## Deliverables

1. `shell_rft/generation/file_counting.py`
2. Updated `shell_rft/generation/__init__.py` with registry
3. Updated `scripts/generate_dataset.py` with working CLI
4. `tests/test_file_counting.py`
5. `tests/test_generate_dataset.py`
6. All existing + new tests pass (`uv run pytest`)
