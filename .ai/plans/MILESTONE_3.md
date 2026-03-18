# Milestone 3: Remaining Task Family Generators

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Status: COMPLETE**

**Goal:** Implement the three remaining task family generators (content_search, topk_by_size, csv_filtering), register them, and verify the dataset generation script produces a balanced multi-family JSONL dataset.

**Architecture:** Three new generator modules follow the same pattern as `file_counting.py`: each exports a `generate_X_examples(n, rng)` function that produces `Example` objects. Each new generator has 3 sub-types selected randomly per example (file_counting has 4). The existing generation registry and dataset script already support multiple generators — we just add entries to `GENERATORS`.

**Tech Stack:** Python 3.11+, standard library only (random, dataclasses), pytest

---

## Design

### Content search generator

**Module**: `shell_rft/generation/content_search.py`

Three sub-types:

| Sub-type | Task example | Expected command | Expected output |
|----------|-------------|-----------------|-----------------|
| Files containing string | "Print the paths of files under logs/ that contain 'ERROR', sorted." | `grep -rl 'ERROR' logs/ \| sort` | Sorted file paths |
| Matching lines from file | "Print the lines from logs/debug.log that contain 'timeout'." | `grep 'timeout' logs/debug.log` | Matching lines |
| Files NOT containing | "Print the paths of files under src/ that do NOT contain 'TODO', sorted." | `grep -rL 'TODO' src/ \| sort` | Sorted file paths |

**Key decisions:**
- Tasks always specify a directory ("under logs/") so output paths use a predictable prefix (`logs/...`), avoiding `./` ambiguity.
- Search terms are simple strings without regex metacharacters: `"ERROR"`, `"TODO"`, `"import"`, `"timeout"`, `"hello"`, `"DEBUG"`.
- Filesystem summary shows only file paths (no content, no sizes) — same format as file_counting.
- Expected output is computed from workspace data using Python substring matching, which is equivalent to `grep` for these simple terms.
- **Known limitation**: If the model uses `grep -rl 'term' .` instead of `grep -rl 'term' dir/`, the output will have `./` prefixed paths which won't match expected output. This is an acceptable false negative — the task text says "under dir/", guiding the model toward the matching format.

### Top-k by size generator

**Module**: `shell_rft/generation/topk_by_size.py`

| Sub-type | Task example | Expected command |
|----------|-------------|-----------------|
| K largest by extension | "Print the 3 largest .log files under app/, largest first." | `find app/ -name '*.log' \| xargs wc -c \| grep -v total \| sort -rn \| head -3 \| awk '{print $2}'` |
| K largest in dir | "Print the 3 largest files under data/, largest first." | `find data/ -type f \| xargs wc -c \| grep -v total \| sort -rn \| head -3 \| awk '{print $2}'` |
| K smallest in dir | "Print the 2 smallest files under logs/, smallest first." | Same pattern with `sort -n` instead of `sort -rn` |

**Key decisions:**
- Workspace files have distinct sizes generated via `rng.sample(range(100, 10001), n_files)`.
- File content is `"x" * (size - 1) + "\n"` — exactly `size` bytes, matching what `wc -c` reports.
- Filesystem summary includes sizes: `- app/logs/debug.log (3000 bytes)`.
- All sub-types require at least 2 matching files (`while True` retry loop).

### CSV filtering generator

**Module**: `shell_rft/generation/csv_filtering.py`

| Sub-type | Task example | Expected command |
|----------|-------------|-----------------|
| Numeric filter | "Print the name values from data/users.csv where usage_pct is greater than 80." | `awk -F',' '$3 > 80 {print $1}' data/users.csv` |
| String filter | "Print the name values from data/users.csv where role is 'admin'." | `awk -F',' '$2 == "admin" {print $1}' data/users.csv` |
| Count by numeric | "How many rows in data/servers.csv have response_ms greater than 500?" | `awk -F',' '$3 > 500' data/servers.csv \| wc -l` |

**Key decisions:**
- Three CSV schemas rotate randomly: users (name/role/usage_pct), servers (host/status/response_ms), jobs (job_id/priority/duration_s).
- Column layout is always: `columns[0]` = identifier (output), `columns[1]` = category (string filter), `columns[2]` = numeric (numeric filter).
- CSV files have **no header row** — awk comparison operators do string comparison on non-numeric values, so a header row would match `$3 > T` conditions. Column names are conveyed via the filesystem summary instead.
- Filesystem summary shows column names: `- data/users.csv (columns: name,role,usage_pct)`.
- Workspace contains only the CSV file (no dummy files).
- Threshold selection guarantees at least one matching row.

### Design constraints

- **Same pattern**: All generators follow the file_counting pattern — `_build_workspace`, `_make_example`, `_SUB_TYPES` list, top-level `generate_X_examples(n, rng)`.
- **Determinism**: All randomness flows through `random.Random` instances.
- **No new deps**: Standard library only.
- **Prompt contract**: All generators use `SYSTEM_PROMPT` and `USER_PROMPT_TEMPLATE`.
- **Correctness**: Expected output is computed from the workspace data, not from a separate calculation.

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `shell_rft/generation/content_search.py` | Create | Generate content-search examples across 3 sub-types |
| `shell_rft/generation/topk_by_size.py` | Create | Generate top-k-by-size examples across 3 sub-types |
| `shell_rft/generation/csv_filtering.py` | Create | Generate CSV filtering examples across 3 sub-types |
| `shell_rft/generation/__init__.py` | Modify | Add 3 new entries to `GENERATORS` registry |
| `tests/test_content_search.py` | Create | Tests for content-search generator |
| `tests/test_topk_by_size.py` | Create | Tests for top-k-by-size generator |
| `tests/test_csv_filtering.py` | Create | Tests for CSV filtering generator |
| `tests/test_generation_registry.py` | Modify | Update to check for all 4 families |

---

## Chunk 1: Content search generator

### Task 1: Content search tests and implementation

**Files:**
- Create: `tests/test_content_search.py`
- Create: `shell_rft/generation/content_search.py`
- Modify: `shell_rft/generation/__init__.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_content_search.py`:

```python
"""Tests for the content-search task family generator."""

import random

from shell_rft.generation.content_search import generate_content_search_examples
from shell_rft.prompts import SYSTEM_PROMPT


def test_generates_requested_count():
    examples = generate_content_search_examples(10, random.Random(42))
    assert len(examples) == 10


def test_deterministic_with_seed():
    a = generate_content_search_examples(5, random.Random(99))
    b = generate_content_search_examples(5, random.Random(99))
    for x, y in zip(a, b):
        assert x.expected_stdout == y.expected_stdout
        assert x.messages == y.messages


def test_example_structure():
    for ex in generate_content_search_examples(20, random.Random(42)):
        assert ex.task_type == "content_search"
        assert len(ex.messages) == 2
        assert len(ex.workspace_spec.files) > 0
        assert ex.expected_stdout.endswith("\n")
        assert len(ex.expected_stdout.strip()) > 0


def test_messages_use_prompt_contract():
    for ex in generate_content_search_examples(5, random.Random(42)):
        assert ex.messages[0] == {"role": "system", "content": SYSTEM_PROMPT}
        assert ex.messages[1]["role"] == "user"
        assert "Task:" in ex.messages[1]["content"]
        assert "Filesystem summary:" in ex.messages[1]["content"]


def test_workspace_files_exist_in_summary():
    for ex in generate_content_search_examples(10, random.Random(42)):
        user_msg = ex.messages[1]["content"]
        for f in ex.workspace_spec.files:
            assert f.path in user_msg


def test_expected_stdout_is_non_empty():
    """All content search results must have non-empty output."""
    for ex in generate_content_search_examples(50, random.Random(42)):
        assert len(ex.expected_stdout.strip()) > 0


def test_all_sub_types_exercised():
    """All 3 sub-types should appear across a reasonable sample."""
    examples = generate_content_search_examples(100, random.Random(42))
    tasks = [ex.messages[1]["content"] for ex in examples]
    # _find_files_containing: "contain '..." but NOT "do NOT contain"
    assert any("contain '" in t and "do NOT" not in t for t in tasks)
    # _grep_lines_from_file: "lines from"
    assert any("lines from" in t for t in tasks)
    # _find_files_not_containing: "do NOT contain"
    assert any("do NOT contain" in t for t in tasks)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_content_search.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'shell_rft.generation.content_search'`

- [ ] **Step 3: Implement the generator**

Create `shell_rft/generation/content_search.py`:

```python
"""Content-search task family generator."""

from __future__ import annotations

import random

from shell_rft.prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE
from shell_rft.schemas import Example, FileSpec, WorkspaceSpec

_DIRS = ["src", "logs", "data", "app", "lib", "tests", "docs", "config"]
_SUBDIRS = ["utils", "core", "api", "auth", "models"]
_EXTENSIONS = [".py", ".txt", ".log", ".csv", ".json"]
_STEMS = [
    "main", "utils", "config", "app", "debug", "error",
    "output", "report", "index", "helper",
]
_SEARCH_TERMS = ["ERROR", "TODO", "import", "timeout", "hello", "DEBUG"]
_CONTENT_POOL = [
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
    "WARNING: disk space low",
    "import sys",
    "return None",
    "DEBUG: request received",
    "ERROR: timeout exceeded",
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
        if rng.random() < 0.3:
            d = f"{d}/{rng.choice(_SUBDIRS)}"
        stem = rng.choice(_STEMS)
        ext = rng.choice(_EXTENSIONS)
        path = f"{d}/{stem}{ext}"

        if path in seen:
            path = f"{d}/{stem}_{i}{ext}"
        if path in seen:
            continue
        seen.add(path)

        n_lines = rng.randint(1, 5)
        content = "\n".join(rng.choices(_CONTENT_POOL, k=n_lines)) + "\n"
        files.append(FileSpec(path=path, content=content))

    return files


def _filesystem_summary(files: list[FileSpec]) -> str:
    return "\n".join(f"- {f.path}" for f in files)


def _make_example(task: str, files: list[FileSpec], expected: str) -> Example:
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
        expected_stdout=expected,
        task_type="content_search",
    )


def _find_files_containing(rng: random.Random) -> Example:
    """List files containing a search term under a directory, sorted."""
    while True:
        files = _build_workspace(rng)
        dirs = sorted({f.path.split("/")[0] for f in files})
        target_dir = rng.choice(dirs)
        terms = list(_SEARCH_TERMS)
        rng.shuffle(terms)
        for term in terms:
            matches = sorted(
                f.path for f in files
                if f.path.startswith(f"{target_dir}/") and term in f.content
            )
            if matches:
                task = (
                    f"Print the paths of files under {target_dir}/ that "
                    f"contain '{term}', one per line, sorted."
                )
                expected = "\n".join(matches) + "\n"
                return _make_example(task, files, expected)


def _grep_lines_from_file(rng: random.Random) -> Example:
    """Print matching lines from a specific file."""
    while True:
        files = _build_workspace(rng)
        candidates = list(files)
        rng.shuffle(candidates)
        for f in candidates:
            lines = [line for line in f.content.split("\n") if line]
            terms = list(_SEARCH_TERMS)
            rng.shuffle(terms)
            for term in terms:
                matching = [line for line in lines if term in line]
                if matching:
                    task = (
                        f"Print the lines from {f.path} that contain '{term}'."
                    )
                    expected = "\n".join(matching) + "\n"
                    return _make_example(task, files, expected)


def _find_files_not_containing(rng: random.Random) -> Example:
    """List files NOT containing a search term under a directory, sorted."""
    while True:
        files = _build_workspace(rng)
        dirs = sorted({f.path.split("/")[0] for f in files})
        target_dir = rng.choice(dirs)
        dir_files = [f for f in files if f.path.startswith(f"{target_dir}/")]
        if len(dir_files) < 2:
            continue
        terms = list(_SEARCH_TERMS)
        rng.shuffle(terms)
        for term in terms:
            containing = [f for f in dir_files if term in f.content]
            not_containing = sorted(
                f.path for f in dir_files if term not in f.content
            )
            # Need both some containing and some not for a useful task.
            if containing and not_containing:
                task = (
                    f"Print the paths of files under {target_dir}/ that do NOT "
                    f"contain '{term}', one per line, sorted."
                )
                expected = "\n".join(not_containing) + "\n"
                return _make_example(task, files, expected)


_SUB_TYPES = [
    _find_files_containing,
    _grep_lines_from_file,
    _find_files_not_containing,
]


def generate_content_search_examples(
    n: int,
    rng: random.Random,
) -> list[Example]:
    """Generate n content-search examples using the given RNG."""
    return [rng.choice(_SUB_TYPES)(rng) for _ in range(n)]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_content_search.py -v`
Expected: All 7 tests PASS

- [ ] **Step 5: Register in GENERATORS**

Update `shell_rft/generation/__init__.py` — add the import and registry entry:

```python
"""Task family generators."""

from __future__ import annotations

import random
from collections.abc import Callable

from shell_rft.generation.content_search import generate_content_search_examples
from shell_rft.generation.file_counting import generate_file_counting_examples
from shell_rft.schemas import Example

GENERATORS: dict[str, Callable[[int, random.Random], list[Example]]] = {
    "file_counting": generate_file_counting_examples,
    "content_search": generate_content_search_examples,
}
```

- [ ] **Step 6: Run all root tests**

Run: `uv run pytest -v`
Expected: All tests PASS (existing + 7 new content_search tests)

---

## Chunk 2: Top-k by size generator

### Task 2: Top-k by size tests and implementation

**Files:**
- Create: `tests/test_topk_by_size.py`
- Create: `shell_rft/generation/topk_by_size.py`
- Modify: `shell_rft/generation/__init__.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_topk_by_size.py`:

```python
"""Tests for the top-k by size task family generator."""

import random

from shell_rft.generation.topk_by_size import generate_topk_by_size_examples
from shell_rft.prompts import SYSTEM_PROMPT


def test_generates_requested_count():
    examples = generate_topk_by_size_examples(10, random.Random(42))
    assert len(examples) == 10


def test_deterministic_with_seed():
    a = generate_topk_by_size_examples(5, random.Random(99))
    b = generate_topk_by_size_examples(5, random.Random(99))
    for x, y in zip(a, b):
        assert x.expected_stdout == y.expected_stdout
        assert x.messages == y.messages


def test_example_structure():
    for ex in generate_topk_by_size_examples(20, random.Random(42)):
        assert ex.task_type == "topk_by_size"
        assert len(ex.messages) == 2
        assert len(ex.workspace_spec.files) > 0
        assert ex.expected_stdout.endswith("\n")
        lines = ex.expected_stdout.strip().split("\n")
        assert len(lines) >= 2  # At least k=2


def test_messages_use_prompt_contract():
    for ex in generate_topk_by_size_examples(5, random.Random(42)):
        assert ex.messages[0] == {"role": "system", "content": SYSTEM_PROMPT}
        assert ex.messages[1]["role"] == "user"
        assert "Task:" in ex.messages[1]["content"]
        assert "Filesystem summary:" in ex.messages[1]["content"]


def test_filesystem_summary_includes_sizes():
    """Top-k summaries must show file sizes."""
    for ex in generate_topk_by_size_examples(10, random.Random(42)):
        user_msg = ex.messages[1]["content"]
        assert "bytes)" in user_msg


def test_expected_paths_exist_in_workspace():
    """Every path in expected output must be a real workspace file."""
    for ex in generate_topk_by_size_examples(30, random.Random(42)):
        ws_paths = {f.path for f in ex.workspace_spec.files}
        for line in ex.expected_stdout.strip().split("\n"):
            assert line in ws_paths, f"{line!r} not in workspace"


def test_all_sub_types_exercised():
    """All 3 sub-types should appear across a reasonable sample."""
    examples = generate_topk_by_size_examples(100, random.Random(42))
    tasks = [ex.messages[1]["content"] for ex in examples]
    # _largest_by_extension: "largest .ext files under" (dot before "files")
    assert any("largest ." in t for t in tasks), "No _largest_by_extension examples"
    # _largest_in_dir: "largest files under" (no extension)
    assert any("largest files under" in t for t in tasks), "No _largest_in_dir examples"
    # _smallest_in_dir: "smallest"
    assert any("smallest" in t for t in tasks), "No _smallest_in_dir examples"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_topk_by_size.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement the generator**

Create `shell_rft/generation/topk_by_size.py`:

```python
"""Top-k by size task family generator."""

from __future__ import annotations

import random

from shell_rft.prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE
from shell_rft.schemas import Example, FileSpec, WorkspaceSpec

_DIRS = ["src", "logs", "data", "app", "lib", "tests", "docs", "config"]
_SUBDIRS = ["utils", "core", "api", "auth", "models"]
_EXTENSIONS = [".py", ".txt", ".log", ".csv", ".json"]
_STEMS = [
    "main", "utils", "config", "app", "debug", "error",
    "output", "report", "index", "helper",
]


def _build_sized_workspace(rng: random.Random) -> list[FileSpec]:
    """Create files with distinct, known sizes."""
    n_dirs = rng.randint(2, 4)
    n_files = rng.randint(8, 15)
    dirs = rng.sample(_DIRS, n_dirs)

    sizes = rng.sample(range(100, 10001), n_files)

    files: list[FileSpec] = []
    seen: set[str] = set()

    for i in range(n_files):
        d = rng.choice(dirs)
        if rng.random() < 0.3:
            d = f"{d}/{rng.choice(_SUBDIRS)}"
        stem = rng.choice(_STEMS)
        ext = rng.choice(_EXTENSIONS)
        path = f"{d}/{stem}{ext}"

        if path in seen:
            path = f"{d}/{stem}_{i}{ext}"
        if path in seen:
            continue
        seen.add(path)

        content = "x" * (sizes[i] - 1) + "\n"
        files.append(FileSpec(path=path, content=content))

    return files


def _file_ext(path: str) -> str:
    """Return the file extension including the dot, e.g. '.py'."""
    dot = path.rfind(".")
    return path[dot:] if dot != -1 else ""


def _filesystem_summary_with_sizes(files: list[FileSpec]) -> str:
    return "\n".join(f"- {f.path} ({len(f.content)} bytes)" for f in files)


def _make_example(task: str, files: list[FileSpec], expected: str) -> Example:
    user_content = USER_PROMPT_TEMPLATE.format(
        task=task,
        filesystem_summary=_filesystem_summary_with_sizes(files),
    )
    return Example(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        workspace_spec=WorkspaceSpec(files=files),
        expected_stdout=expected,
        task_type="topk_by_size",
    )


def _largest_by_extension(rng: random.Random) -> Example:
    """K largest files of a given extension under a directory."""
    while True:
        files = _build_sized_workspace(rng)
        dirs = sorted({f.path.split("/")[0] for f in files})
        target_dir = rng.choice(dirs)
        dir_files = [f for f in files if f.path.startswith(f"{target_dir}/")]
        exts = list({_file_ext(f.path) for f in dir_files})
        rng.shuffle(exts)
        for ext in exts:
            matching = [f for f in dir_files if f.path.endswith(ext)]
            if len(matching) >= 2:
                matching.sort(key=lambda f: len(f.content), reverse=True)
                k = rng.randint(2, min(4, len(matching)))
                top_k = matching[:k]
                task = (
                    f"Print the {k} largest {ext} files under {target_dir}/, "
                    f"one per line, largest first."
                )
                expected = "\n".join(f.path for f in top_k) + "\n"
                return _make_example(task, files, expected)


def _largest_in_dir(rng: random.Random) -> Example:
    """K largest files (any extension) under a directory."""
    while True:
        files = _build_sized_workspace(rng)
        dirs = sorted({f.path.split("/")[0] for f in files})
        target_dir = rng.choice(dirs)
        matching = [f for f in files if f.path.startswith(f"{target_dir}/")]
        if len(matching) >= 2:
            matching.sort(key=lambda f: len(f.content), reverse=True)
            k = rng.randint(2, min(4, len(matching)))
            top_k = matching[:k]
            task = (
                f"Print the {k} largest files under {target_dir}/, "
                f"one per line, largest first."
            )
            expected = "\n".join(f.path for f in top_k) + "\n"
            return _make_example(task, files, expected)


def _smallest_in_dir(rng: random.Random) -> Example:
    """K smallest files under a directory."""
    while True:
        files = _build_sized_workspace(rng)
        dirs = sorted({f.path.split("/")[0] for f in files})
        target_dir = rng.choice(dirs)
        matching = [f for f in files if f.path.startswith(f"{target_dir}/")]
        if len(matching) >= 2:
            matching.sort(key=lambda f: len(f.content))
            k = rng.randint(2, min(4, len(matching)))
            bottom_k = matching[:k]
            task = (
                f"Print the {k} smallest files under {target_dir}/, "
                f"one per line, smallest first."
            )
            expected = "\n".join(f.path for f in bottom_k) + "\n"
            return _make_example(task, files, expected)


_SUB_TYPES = [
    _largest_by_extension,
    _largest_in_dir,
    _smallest_in_dir,
]


def generate_topk_by_size_examples(
    n: int,
    rng: random.Random,
) -> list[Example]:
    """Generate n top-k-by-size examples using the given RNG."""
    return [rng.choice(_SUB_TYPES)(rng) for _ in range(n)]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_topk_by_size.py -v`
Expected: All 7 tests PASS

- [ ] **Step 5: Register in GENERATORS**

Update `shell_rft/generation/__init__.py` — add the import and registry entry:

```python
"""Task family generators."""

from __future__ import annotations

import random
from collections.abc import Callable

from shell_rft.generation.content_search import generate_content_search_examples
from shell_rft.generation.file_counting import generate_file_counting_examples
from shell_rft.generation.topk_by_size import generate_topk_by_size_examples
from shell_rft.schemas import Example

GENERATORS: dict[str, Callable[[int, random.Random], list[Example]]] = {
    "file_counting": generate_file_counting_examples,
    "content_search": generate_content_search_examples,
    "topk_by_size": generate_topk_by_size_examples,
}
```

- [ ] **Step 6: Run all root tests**

Run: `uv run pytest -v`
Expected: All tests PASS

---

## Chunk 3: CSV filtering, registry, and verification

### Task 3: CSV filtering tests and implementation

**Files:**
- Create: `tests/test_csv_filtering.py`
- Create: `shell_rft/generation/csv_filtering.py`
- Modify: `shell_rft/generation/__init__.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_csv_filtering.py`:

```python
"""Tests for the CSV filtering task family generator."""

import random

from shell_rft.generation.csv_filtering import generate_csv_filtering_examples
from shell_rft.prompts import SYSTEM_PROMPT


def test_generates_requested_count():
    examples = generate_csv_filtering_examples(10, random.Random(42))
    assert len(examples) == 10


def test_deterministic_with_seed():
    a = generate_csv_filtering_examples(5, random.Random(99))
    b = generate_csv_filtering_examples(5, random.Random(99))
    for x, y in zip(a, b):
        assert x.expected_stdout == y.expected_stdout
        assert x.messages == y.messages


def test_example_structure():
    for ex in generate_csv_filtering_examples(20, random.Random(42)):
        assert ex.task_type == "csv_filtering"
        assert len(ex.messages) == 2
        assert len(ex.workspace_spec.files) > 0
        assert ex.expected_stdout.endswith("\n")
        assert len(ex.expected_stdout.strip()) > 0


def test_messages_use_prompt_contract():
    for ex in generate_csv_filtering_examples(5, random.Random(42)):
        assert ex.messages[0] == {"role": "system", "content": SYSTEM_PROMPT}
        assert ex.messages[1]["role"] == "user"
        assert "Task:" in ex.messages[1]["content"]
        assert "Filesystem summary:" in ex.messages[1]["content"]


def test_filesystem_summary_shows_columns():
    """CSV summaries must include column names."""
    for ex in generate_csv_filtering_examples(10, random.Random(42)):
        user_msg = ex.messages[1]["content"]
        assert "columns:" in user_msg


def test_workspace_has_csv_file():
    """Workspace must contain at least one CSV file."""
    for ex in generate_csv_filtering_examples(20, random.Random(42)):
        csv_files = [f for f in ex.workspace_spec.files if f.path.endswith(".csv")]
        assert len(csv_files) >= 1


def test_expected_stdout_is_non_empty():
    """All CSV filtering results must have non-empty output."""
    for ex in generate_csv_filtering_examples(50, random.Random(42)):
        assert len(ex.expected_stdout.strip()) > 0


def test_all_sub_types_exercised():
    """All 3 sub-types should appear across a reasonable sample."""
    examples = generate_csv_filtering_examples(100, random.Random(42))
    tasks = [ex.messages[1]["content"] for ex in examples]
    # _numeric_filter: "Print ... greater than ..."
    assert any("Print" in t and "greater than" in t for t in tasks)
    # _string_filter: "is '..."
    assert any("is '" in t for t in tasks)
    # _count_by_numeric: "How many ..."
    assert any("How many" in t for t in tasks)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_csv_filtering.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement the generator**

Create `shell_rft/generation/csv_filtering.py`:

```python
"""CSV filtering task family generator."""

from __future__ import annotations

import random

from shell_rft.prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE
from shell_rft.schemas import Example, FileSpec, WorkspaceSpec

_CSV_SCHEMAS = [
    {
        "path": "data/users.csv",
        "columns": ["name", "role", "usage_pct"],
        "names": [
            "alice", "bob", "charlie", "diana",
            "eve", "frank", "grace", "heidi",
        ],
        "categories": ["admin", "user", "viewer"],
        "numeric_range": (10, 99),
    },
    {
        "path": "data/servers.csv",
        "columns": ["host", "status", "response_ms"],
        "names": [
            "web01", "web02", "db01", "db02",
            "cache01", "api01", "api02", "worker01",
        ],
        "categories": ["up", "down", "degraded"],
        "numeric_range": (50, 2000),
    },
    {
        "path": "data/jobs.csv",
        "columns": ["job_id", "priority", "duration_s"],
        "names": [
            "job_001", "job_002", "job_003", "job_004",
            "job_005", "job_006", "job_007", "job_008",
        ],
        "categories": ["high", "medium", "low"],
        "numeric_range": (1, 500),
    },
]


def _generate_csv_data(
    schema: dict, rng: random.Random,
) -> tuple[str, list[list[str]]]:
    """Generate CSV content. Returns (content_string, data_rows_as_lists)."""
    n_rows = rng.randint(5, 12)
    columns = schema["columns"]
    header = ",".join(columns)

    names = list(schema["names"])
    rng.shuffle(names)
    names = names[:n_rows]
    # Pad if not enough unique names.
    while len(names) < n_rows:
        names.append(f"{rng.choice(schema['names'])}_{len(names)}")

    rows: list[list[str]] = []
    lines = [header]
    for name in names:
        category = rng.choice(schema["categories"])
        numeric = rng.randint(*schema["numeric_range"])
        row = [name, category, str(numeric)]
        rows.append(row)
        lines.append(",".join(row))

    content = "\n".join(lines) + "\n"
    return content, rows


def _pick_threshold(values: list[int], rng: random.Random) -> int:
    """Pick a threshold that guarantees at least one value exceeds it."""
    sorted_vals = sorted(set(values))
    split_idx = rng.randint(0, len(sorted_vals) - 1)
    return sorted_vals[split_idx] - 1


def _filesystem_summary(files: list[FileSpec]) -> str:
    summaries = []
    for f in files:
        if f.path.endswith(".csv"):
            header_line = f.content.split("\n")[0]
            summaries.append(f"- {f.path} (columns: {header_line})")
        else:
            summaries.append(f"- {f.path}")
    return "\n".join(summaries)


def _make_example(task: str, files: list[FileSpec], expected: str) -> Example:
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
        expected_stdout=expected,
        task_type="csv_filtering",
    )


def _numeric_filter(rng: random.Random) -> Example:
    """Print column values where a numeric column exceeds a threshold."""
    schema = rng.choice(_CSV_SCHEMAS)
    content, rows = _generate_csv_data(schema, rng)
    columns = schema["columns"]
    numeric_col = columns[2]
    output_col = columns[0]

    values = [int(row[2]) for row in rows]
    threshold = _pick_threshold(values, rng)
    matching = [row[0] for row in rows if int(row[2]) > threshold]

    path = schema["path"]
    task = (
        f"Print the {output_col} values from {path} where "
        f"{numeric_col} is greater than {threshold}."
    )
    expected = "\n".join(matching) + "\n"
    files = [FileSpec(path=path, content=content)]
    return _make_example(task, files, expected)


def _string_filter(rng: random.Random) -> Example:
    """Print column values where a string column matches a value."""
    schema = rng.choice(_CSV_SCHEMAS)
    content, rows = _generate_csv_data(schema, rng)
    columns = schema["columns"]
    string_col = columns[1]
    output_col = columns[0]

    # Pick a category value that appears in the data.
    present_categories = list({row[1] for row in rows})
    filter_value = rng.choice(present_categories)
    matching = [row[0] for row in rows if row[1] == filter_value]

    path = schema["path"]
    task = (
        f"Print the {output_col} values from {path} where "
        f"{string_col} is '{filter_value}'."
    )
    expected = "\n".join(matching) + "\n"
    files = [FileSpec(path=path, content=content)]
    return _make_example(task, files, expected)


def _count_by_numeric(rng: random.Random) -> Example:
    """Count rows where a numeric column exceeds a threshold."""
    schema = rng.choice(_CSV_SCHEMAS)
    content, rows = _generate_csv_data(schema, rng)
    columns = schema["columns"]
    numeric_col = columns[2]

    values = [int(row[2]) for row in rows]
    threshold = _pick_threshold(values, rng)
    count = sum(1 for row in rows if int(row[2]) > threshold)

    path = schema["path"]
    task = (
        f"How many rows in {path} have {numeric_col} greater than {threshold}?"
    )
    expected = f"{count}\n"
    files = [FileSpec(path=path, content=content)]
    return _make_example(task, files, expected)


_SUB_TYPES = [
    _numeric_filter,
    _string_filter,
    _count_by_numeric,
]


def generate_csv_filtering_examples(
    n: int,
    rng: random.Random,
) -> list[Example]:
    """Generate n csv-filtering examples using the given RNG."""
    return [rng.choice(_SUB_TYPES)(rng) for _ in range(n)]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_csv_filtering.py -v`
Expected: All 8 tests PASS

- [ ] **Step 5: Register in GENERATORS**

Update `shell_rft/generation/__init__.py` to its final form:

```python
"""Task family generators."""

from __future__ import annotations

import random
from collections.abc import Callable

from shell_rft.generation.content_search import generate_content_search_examples
from shell_rft.generation.csv_filtering import generate_csv_filtering_examples
from shell_rft.generation.file_counting import generate_file_counting_examples
from shell_rft.generation.topk_by_size import generate_topk_by_size_examples
from shell_rft.schemas import Example

GENERATORS: dict[str, Callable[[int, random.Random], list[Example]]] = {
    "file_counting": generate_file_counting_examples,
    "content_search": generate_content_search_examples,
    "topk_by_size": generate_topk_by_size_examples,
    "csv_filtering": generate_csv_filtering_examples,
}
```

### Task 4: Registry update and final verification

**Files:**
- Modify: `tests/test_generation_registry.py`

- [ ] **Step 1: Update the registry test**

Replace contents of `tests/test_generation_registry.py`:

```python
"""Tests for the generation registry."""

from shell_rft.generation import GENERATORS


def test_registry_contains_all_families():
    expected = {"file_counting", "content_search", "topk_by_size", "csv_filtering"}
    assert set(GENERATORS.keys()) == expected


def test_registry_values_are_callable():
    for gen_fn in GENERATORS.values():
        assert callable(gen_fn)
```

- [ ] **Step 2: Run all root tests**

Run: `uv run pytest -v`
Expected: All tests PASS (3 smoke + 7 file_counting + 7 content_search + 7 topk_by_size + 8 csv_filtering + 2 registry + 2 dataset = 36 tests)

- [ ] **Step 3: Run evaluator tests to confirm nothing broken**

Run: `cd evaluators/shell_rft_v0 && uv run pytest -vs`
Expected: All evaluator tests PASS (unchanged)

- [ ] **Step 4: Generate a multi-family dataset and spot-check**

```bash
uv run python scripts/generate_dataset.py --train-n 20 --val-n 5 --test-n 5 --output-dir /tmp/shell-rft-m3
```

Verify all 4 task types appear:

```bash
cat /tmp/shell-rft-m3/train.jsonl | python3 -c "
import json, sys, collections
counts = collections.Counter()
for line in sys.stdin:
    row = json.loads(line)
    counts[row['task_type']] += 1
print(dict(counts))
"
```

Expected: `{'file_counting': 5, 'content_search': 5, 'topk_by_size': 5, 'csv_filtering': 5}`

Spot-check one row from each task type:

```bash
head -1 /tmp/shell-rft-m3/train.jsonl | python3 -m json.tool | head -20
```

- [ ] **Step 5: Update milestone status**

Change this file's status from `NOT STARTED` to `COMPLETE`.

---

## Deliverables

1. `shell_rft/generation/content_search.py` — 3 sub-types: files containing, lines from file, files NOT containing
2. `shell_rft/generation/topk_by_size.py` — 3 sub-types: largest by extension, largest in dir, smallest in dir
3. `shell_rft/generation/csv_filtering.py` — 3 sub-types: numeric filter, string filter, count by numeric
4. Updated `shell_rft/generation/__init__.py` with all 4 generators registered
5. `tests/test_content_search.py` — 7 tests
6. `tests/test_topk_by_size.py` — 7 tests
7. `tests/test_csv_filtering.py` — 8 tests
8. Updated `tests/test_generation_registry.py` — checks for all 4 families
9. All existing + new tests pass (`uv run pytest`)
10. Multi-family dataset generation verified
