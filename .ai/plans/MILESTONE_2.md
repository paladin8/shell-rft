# Milestone 2: Local Evaluator with Sandbox and Tests

**Status: COMPLETE**

**Goal:** Implement the four evaluator modules (`normalize.py`, `policy.py`, `sandbox.py`, `main.py`) so the evaluator can score model responses end-to-end, and cover them with thorough tests.

**Architecture:** The evaluator is a self-contained package in `evaluators/shell_rft_v0/`. It receives a model's assistant message plus ground-truth fields (`workspace_spec`, `expected_stdout`, `task_type`), extracts a single command, validates it against a policy allowlist, materializes the workspace in a temp directory, executes the command in a sandboxed subprocess, normalizes the output, and compares against expected. Reward is binary: 1.0 on exact match, 0.0 otherwise.

**Tech Stack:** Python 3.11+, `fireworks-ai[reward-kit]`, `subprocess`, `tempfile`, `shlex`, pytest

---

## What's already done

- `main.py` — `@reward_function` decorator and signature, raises `NotImplementedError`
- `sandbox.py` — `materialize_workspace` and `run_command` signatures, raise `NotImplementedError`
- `policy.py` — `ALLOWED_COMMANDS` frozenset defined, `validate_command` and `extract_single_command` stubs
- `normalize.py` — `normalize_stdout` stub
- `tests/test_evaluator_smoke.py` — 2 tests for `ALLOWED_COMMANDS`

---

## Design

### `normalize.py` — stdout normalization

Simple normalization for exact-match comparison:

1. Strip trailing whitespace from each line
2. Strip trailing newlines from the full output
3. Re-add a single trailing newline (canonical form)

This keeps it simple for v0. All task families produce output where line order matters, so no sorting.

```python
def normalize_stdout(stdout: str, task_type: str | None = None) -> str:
```

### `policy.py` — command extraction and validation

**`extract_single_command(text)`**

Extracts exactly one shell command from model output:

1. Strip leading/trailing whitespace
2. If the text contains a markdown fenced code block, extract its contents
3. After extraction, if there are multiple non-empty lines, return `None` (model gave multiple commands or extra explanation)
4. Return the single line, or `None` if empty

**`validate_command(command)`**

Validates a command string against the sandbox policy. Returns a violation reason string, or `None` if the command is allowed.

Checks (in order):
1. **Shell control operators**: reject `&&`, `||` — these create multiple independent commands
2. **Semicolons**: reject `;` — also creates multiple commands. Known limitation: this also rejects valid `find -exec {} \;` commands, which is an acceptable false negative for v0's conservative policy.
3. **Subshells**: reject `$(`, backticks — prevents arbitrary nesting
4. **Pipeline splitting**: split by `|` to get pipeline stages
5. **Command allowlist**: for each pipeline stage, extract the first token (via `shlex.split`) and check it's in `ALLOWED_COMMANDS`
6. **Token-level checks** (per pipeline stage, on `shlex.split` tokens): reject redirections (`>`, `>>`, `<` as standalone tokens), absolute paths (tokens starting with `/`), and path traversal (`..` segments in any token)

Pipes (`|`) are allowed — `find ... | wc -l` is a single pipeline, not multiple commands.

Redirection, absolute path, and path traversal checks operate on parsed tokens (not raw string matching) so that operators inside quoted strings (e.g., `awk '{print ($1 > 80)}'`) are not falsely rejected.

### `sandbox.py` — workspace materialization and execution

**`materialize_workspace(workspace_spec, workdir)`**

- Iterates `workspace_spec["files"]`
- Creates parent directories with `os.makedirs`
- Writes each file's content

**`run_command(command, cwd, timeout_s=2.0)`**

- Executes via `subprocess.run` with `shell=True` (needed for pipes)
- Sets `timeout` and `cwd`
- Captures stdout and stderr
- On timeout, returns exit code 124 (conventional timeout exit code)
- Returns `(exit_code, stdout, stderr)` tuple

### `main.py` — evaluator entry point

Wires the pieces together:

1. Wrap entire body in `try/except Exception` — evaluator must never crash; any unexpected error returns `{"score": 0.0, "reason": "internal error"}`
2. Get the last assistant message from `messages`
3. `extract_single_command` — if `None`, return `{"score": 0.0, "reason": "no single command"}`
4. `validate_command` — if violation, return `{"score": 0.0, "reason": violation}`
5. Create a temp directory, materialize workspace
6. `run_command` — if non-zero exit, return `{"score": 0.0, "reason": "execution failed"}`
7. `normalize_stdout` on both actual and expected
8. Compare — return `{"score": 1.0}` or `{"score": 0.0, "reason": "output mismatch"}`
9. Clean up temp directory in a `finally` block

### Design constraints

- **Self-contained**: The evaluator dir must not import from `shell_rft.*`. Fireworks Build SDK packages this directory independently.
- **Conservative**: False positives (rewarding wrong answers) are worse than false negatives. When in doubt, score 0.0.
- **No network**: Policy prevents network commands; subprocess doesn't need network restrictions.
- **Deterministic scoring**: Same input always produces same score.

### Out of scope

- Resource limits (cgroups, rlimit) — not needed for toy workspaces
- Partial credit — binary only for v0
- Remote deployment to Fireworks — milestone 5
- Logging/observability — the evaluator returns `reason` strings for debugging, but aggregate metrics (parse-success rate, policy-violation rate, etc.) are deferred to a later milestone

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `evaluators/shell_rft_v0/normalize.py` | Modify | Strip/canonicalize stdout for comparison |
| `evaluators/shell_rft_v0/policy.py` | Modify | Extract single command from model output, validate against allowlist |
| `evaluators/shell_rft_v0/sandbox.py` | Modify | Materialize workspace files, execute command in subprocess |
| `evaluators/shell_rft_v0/main.py` | Modify | Wire all pieces into `@reward_function` entry point |
| `evaluators/shell_rft_v0/tests/test_normalize.py` | Create | Tests for stdout normalization |
| `evaluators/shell_rft_v0/tests/test_policy.py` | Create | Tests for command extraction and validation |
| `evaluators/shell_rft_v0/tests/test_sandbox.py` | Create | Tests for workspace materialization and command execution |
| `evaluators/shell_rft_v0/tests/test_evaluate.py` | Create | End-to-end tests for the reward function |

---

## Chunk 1: normalize and policy

### Task 1: Stdout normalization

**Files:**
- Create: `evaluators/shell_rft_v0/tests/test_normalize.py`
- Modify: `evaluators/shell_rft_v0/normalize.py`

- [ ] **Step 1: Write the failing tests**

Create `evaluators/shell_rft_v0/tests/test_normalize.py`:

```python
"""Tests for stdout normalization."""

from shell_rft_v0.normalize import normalize_stdout


def test_strips_trailing_whitespace_per_line():
    assert normalize_stdout("hello  \nworld \n") == "hello\nworld\n"


def test_strips_trailing_newlines():
    assert normalize_stdout("42\n\n\n") == "42\n"


def test_adds_trailing_newline_if_missing():
    assert normalize_stdout("42") == "42\n"


def test_empty_string():
    assert normalize_stdout("") == "\n"


def test_preserves_internal_whitespace():
    assert normalize_stdout("hello   world\n") == "hello   world\n"


def test_multiple_lines():
    assert normalize_stdout("a.txt\nb.txt\nc.txt\n") == "a.txt\nb.txt\nc.txt\n"


def test_task_type_ignored_for_now():
    """task_type param is accepted but doesn't change behavior in v0."""
    result = normalize_stdout("42\n", task_type="file_counting")
    assert result == "42\n"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd evaluators/shell_rft_v0 && uv run pytest tests/test_normalize.py -v`
Expected: FAIL with `NotImplementedError`

- [ ] **Step 3: Implement normalize_stdout**

Replace contents of `evaluators/shell_rft_v0/normalize.py`:

```python
"""Stdout normalization for comparison."""


def normalize_stdout(stdout: str, task_type: str | None = None) -> str:
    """Normalize stdout for comparison.

    Strips trailing whitespace from each line, collapses trailing newlines,
    and ensures exactly one trailing newline.
    """
    lines = stdout.split("\n")
    # Remove trailing empty strings from split (trailing newlines)
    while lines and lines[-1] == "":
        lines.pop()
    # Strip trailing whitespace from each line
    lines = [line.rstrip() for line in lines]
    # Re-join with single trailing newline
    return "\n".join(lines) + "\n"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd evaluators/shell_rft_v0 && uv run pytest tests/test_normalize.py -v`
Expected: All 7 tests PASS

- [ ] **Step 5: Commit**

```bash
git add evaluators/shell_rft_v0/normalize.py evaluators/shell_rft_v0/tests/test_normalize.py
git commit -m "feat(evaluator): implement stdout normalization"
```

### Task 2: Command extraction

**Files:**
- Create: `evaluators/shell_rft_v0/tests/test_policy.py`
- Modify: `evaluators/shell_rft_v0/policy.py`

- [ ] **Step 1: Write the failing tests for extract_single_command**

Create `evaluators/shell_rft_v0/tests/test_policy.py`:

```python
"""Tests for command extraction and validation."""

from shell_rft_v0.policy import extract_single_command, validate_command


# --- extract_single_command ---


def test_extract_plain_command():
    assert extract_single_command("find . -name '*.py' | wc -l") == "find . -name '*.py' | wc -l"


def test_extract_strips_whitespace():
    assert extract_single_command("  find . -type f  \n") == "find . -type f"


def test_extract_from_markdown_code_block():
    text = "```\nfind . -name '*.log'\n```"
    assert extract_single_command(text) == "find . -name '*.log'"


def test_extract_from_markdown_code_block_with_language():
    text = "```bash\ngrep -r 'ERROR' logs/\n```"
    assert extract_single_command(text) == "grep -r 'ERROR' logs/"


def test_extract_rejects_multiple_lines():
    text = "find . -name '*.py'\nwc -l"
    assert extract_single_command(text) is None


def test_extract_rejects_empty():
    assert extract_single_command("") is None
    assert extract_single_command("   \n  ") is None


def test_extract_rejects_multiple_commands_in_code_block():
    text = "```\nfind . -type f\nwc -l\n```"
    assert extract_single_command(text) is None


def test_extract_ignores_surrounding_explanation():
    """If there's text outside a code block, only the code block content matters."""
    text = "Here's the command:\n```\nfind . -type f | wc -l\n```\nThis counts files."
    assert extract_single_command(text) == "find . -type f | wc -l"


def test_extract_multiple_code_blocks_uses_first():
    """If model emits multiple code blocks, use the first one."""
    text = "Option A:\n```\nfind . -type f | wc -l\n```\nOption B:\n```\nls | wc -l\n```"
    assert extract_single_command(text) == "find . -type f | wc -l"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd evaluators/shell_rft_v0 && uv run pytest tests/test_policy.py -v -k extract`
Expected: FAIL with `NotImplementedError`

- [ ] **Step 3: Implement extract_single_command**

Update `evaluators/shell_rft_v0/policy.py` — replace the `extract_single_command` stub:

```python
import re

def extract_single_command(text: str) -> str | None:
    """Extract a single shell command from model output. Returns None if not exactly one."""
    # Try to extract from a markdown fenced code block first.
    fence_match = re.search(r"```(?:\w*)\n(.*?)```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1)

    # Split into non-empty lines.
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]

    if len(lines) != 1:
        return None

    return lines[0]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd evaluators/shell_rft_v0 && uv run pytest tests/test_policy.py -v -k extract`
Expected: All 9 extraction tests PASS

- [ ] **Step 5: Commit**

```bash
git add evaluators/shell_rft_v0/policy.py evaluators/shell_rft_v0/tests/test_policy.py
git commit -m "feat(evaluator): implement command extraction from model output"
```

### Task 3: Command validation

**Files:**
- Modify: `evaluators/shell_rft_v0/tests/test_policy.py` (append)
- Modify: `evaluators/shell_rft_v0/policy.py`

- [ ] **Step 1: Write the failing tests for validate_command**

Append to `evaluators/shell_rft_v0/tests/test_policy.py`:

```python
# --- validate_command ---


def test_validate_simple_allowed():
    assert validate_command("find . -type f") is None


def test_validate_pipeline_allowed():
    assert validate_command("find . -name '*.py' | wc -l") is None


def test_validate_multi_pipe_allowed():
    assert validate_command("find . -type f | sort | head -5") is None


def test_validate_rejects_disallowed_command():
    reason = validate_command("rm -rf /")
    assert reason is not None
    assert "rm" in reason


def test_validate_rejects_disallowed_in_pipeline():
    reason = validate_command("find . | rm")
    assert reason is not None


def test_validate_rejects_semicolon():
    reason = validate_command("find . -type f; rm -rf /")
    assert reason is not None


def test_validate_rejects_and_operator():
    reason = validate_command("find . && rm -rf /")
    assert reason is not None


def test_validate_rejects_or_operator():
    reason = validate_command("find . || echo fail")
    assert reason is not None


def test_validate_rejects_subshell():
    reason = validate_command("find . -exec $(rm) \\;")
    assert reason is not None


def test_validate_rejects_backtick_subshell():
    reason = validate_command("find . -name `echo test`")
    assert reason is not None


def test_validate_rejects_redirect():
    reason = validate_command("find . > output.txt")
    assert reason is not None


def test_validate_rejects_append_redirect():
    reason = validate_command("find . >> output.txt")
    assert reason is not None


def test_validate_rejects_input_redirect():
    reason = validate_command("sort < input.txt")
    assert reason is not None


def test_validate_rejects_absolute_path():
    reason = validate_command("cat /etc/passwd")
    assert reason is not None


def test_validate_rejects_path_traversal():
    reason = validate_command("cat ../../../etc/passwd")
    assert reason is not None


def test_validate_allows_relative_paths():
    assert validate_command("find src/ -name '*.py'") is None


def test_validate_allows_grep_with_flags():
    assert validate_command("grep -rl 'ERROR' logs/") is None


def test_validate_allows_awk_program():
    assert validate_command("awk '{print $1}' data.csv") is None


def test_validate_allows_awk_with_comparison_operator():
    """Redirection check must not reject > inside quoted awk programs."""
    assert validate_command("awk -F',' '$2 > 80 {print $1}' data.csv") is None


def test_validate_rejects_find_exec_semicolon():
    """Known limitation: find -exec {} \\; is rejected because of the ; check.
    This is an acceptable false negative for v0's conservative policy."""
    reason = validate_command("find . -exec cat {} \\;")
    assert reason is not None


def test_validate_rejects_bare_dotdot():
    """Path traversal without a slash must still be caught."""
    reason = validate_command("cat ..")
    assert reason is not None


def test_validate_rejects_dotdot_in_relative_path():
    reason = validate_command("cat foo/../../../etc/passwd")
    assert reason is not None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd evaluators/shell_rft_v0 && uv run pytest tests/test_policy.py -v -k validate`
Expected: FAIL with `NotImplementedError`

- [ ] **Step 3: Implement validate_command**

Replace the `validate_command` stub in `evaluators/shell_rft_v0/policy.py`:

```python
import shlex

def validate_command(command: str) -> str | None:
    """Check command against policy. Returns violation reason or None if OK."""
    # Check for shell control operators (multiple independent commands).
    # Note: raw string checks will false-reject operators inside quoted strings
    # (e.g., awk -F';'). Accepted limitation for v0's conservative policy.
    for op in ("&&", "||"):
        if op in command:
            return f"shell operator '{op}' is not allowed"
    if ";" in command:
        return "shell operator ';' is not allowed"

    # Check for subshells.
    if "$(" in command or "`" in command:
        return "subshells are not allowed"

    # Split pipeline stages and check each command.
    stages = command.split("|")
    for stage in stages:
        stage = stage.strip()
        if not stage:
            return "empty pipeline stage"
        try:
            tokens = shlex.split(stage)
        except ValueError:
            return "malformed command"

        cmd_name = tokens[0]
        if cmd_name not in ALLOWED_COMMANDS:
            return f"command '{cmd_name}' is not allowed"

        # Check tokens for redirections, absolute paths, and traversal.
        for token in tokens:
            if token in (">", ">>", "<"):
                return f"redirection '{token}' is not allowed"
            if token.startswith("/"):
                return f"absolute path '{token}' is not allowed"
            if ".." in token.split("/"):
                return f"path traversal in '{token}' is not allowed"

    return None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd evaluators/shell_rft_v0 && uv run pytest tests/test_policy.py -v`
Expected: All 31 tests PASS (9 extraction + 22 validation)

- [ ] **Step 5: Commit**

```bash
git add evaluators/shell_rft_v0/policy.py evaluators/shell_rft_v0/tests/test_policy.py
git commit -m "feat(evaluator): implement command validation against policy"
```

---

## Chunk 2: sandbox and main

### Task 4: Workspace materialization and command execution

**Files:**
- Create: `evaluators/shell_rft_v0/tests/test_sandbox.py`
- Modify: `evaluators/shell_rft_v0/sandbox.py`

- [ ] **Step 1: Write the failing tests**

Create `evaluators/shell_rft_v0/tests/test_sandbox.py`:

```python
"""Tests for workspace materialization and sandboxed execution."""

import os
import tempfile

from shell_rft_v0.sandbox import materialize_workspace, run_command


class TestMaterializeWorkspace:
    def test_creates_files(self):
        spec = {
            "files": [
                {"path": "src/main.py", "content": "print('hello')\n"},
                {"path": "data/test.csv", "content": "a,b\n1,2\n"},
            ]
        }
        with tempfile.TemporaryDirectory() as workdir:
            materialize_workspace(spec, workdir)
            main_path = os.path.join(workdir, "src", "main.py")
            assert os.path.exists(main_path)
            with open(main_path) as f:
                assert f.read() == "print('hello')\n"

            csv_path = os.path.join(workdir, "data", "test.csv")
            assert os.path.exists(csv_path)
            with open(csv_path) as f:
                assert f.read() == "a,b\n1,2\n"

    def test_creates_nested_directories(self):
        spec = {
            "files": [
                {"path": "a/b/c/deep.txt", "content": "deep\n"},
            ]
        }
        with tempfile.TemporaryDirectory() as workdir:
            materialize_workspace(spec, workdir)
            assert os.path.exists(os.path.join(workdir, "a", "b", "c", "deep.txt"))

    def test_creates_root_level_file(self):
        spec = {
            "files": [
                {"path": "readme.txt", "content": "hello\n"},
            ]
        }
        with tempfile.TemporaryDirectory() as workdir:
            materialize_workspace(spec, workdir)
            path = os.path.join(workdir, "readme.txt")
            assert os.path.exists(path)
            with open(path) as f:
                assert f.read() == "hello\n"

    def test_empty_workspace(self):
        spec = {"files": []}
        with tempfile.TemporaryDirectory() as workdir:
            materialize_workspace(spec, workdir)
            # Should not error; workdir exists but is empty.
            assert os.path.isdir(workdir)


class TestRunCommand:
    def test_simple_echo(self):
        with tempfile.TemporaryDirectory() as workdir:
            exit_code, stdout, stderr = run_command("echo hello", workdir)
            assert exit_code == 0
            assert stdout == "hello\n"

    def test_command_respects_cwd(self):
        with tempfile.TemporaryDirectory() as workdir:
            # Create a file and find it.
            with open(os.path.join(workdir, "test.txt"), "w") as f:
                f.write("content\n")
            exit_code, stdout, _ = run_command("find . -name 'test.txt'", workdir)
            assert exit_code == 0
            assert "test.txt" in stdout

    def test_failed_command_returns_nonzero(self):
        with tempfile.TemporaryDirectory() as workdir:
            exit_code, _, _ = run_command("false", workdir)
            assert exit_code != 0

    def test_timeout_returns_124(self):
        with tempfile.TemporaryDirectory() as workdir:
            exit_code, _, _ = run_command("sleep 10", workdir, timeout_s=0.5)
            assert exit_code == 124

    def test_pipeline(self):
        with tempfile.TemporaryDirectory() as workdir:
            for name in ["a.py", "b.py", "c.txt"]:
                with open(os.path.join(workdir, name), "w") as f:
                    f.write("x\n")
            exit_code, stdout, _ = run_command(
                "find . -name '*.py' | wc -l", workdir
            )
            assert exit_code == 0
            assert stdout.strip() == "2"

    def test_stderr_captured(self):
        with tempfile.TemporaryDirectory() as workdir:
            exit_code, stdout, stderr = run_command(
                "echo err >&2", workdir
            )
            assert exit_code == 0
            assert stderr.strip() == "err"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd evaluators/shell_rft_v0 && uv run pytest tests/test_sandbox.py -v`
Expected: FAIL with `NotImplementedError`

- [ ] **Step 3: Implement sandbox.py**

Replace contents of `evaluators/shell_rft_v0/sandbox.py`:

```python
"""Workspace materialization and sandboxed command execution."""

import os
import subprocess


def materialize_workspace(workspace_spec: dict, workdir: str) -> None:
    """Write workspace_spec files into workdir."""
    for file_entry in workspace_spec.get("files", []):
        file_path = os.path.join(workdir, file_entry["path"])
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as f:
            f.write(file_entry["content"])


def run_command(
    command: str, cwd: str, timeout_s: float = 2.0
) -> tuple[int, str, str]:
    """Run a shell command in cwd. Returns (exit_code, stdout, stderr)."""
    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        return (result.returncode, result.stdout, result.stderr)
    except subprocess.TimeoutExpired:
        return (124, "", "timeout")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd evaluators/shell_rft_v0 && uv run pytest tests/test_sandbox.py -v`
Expected: All 10 tests PASS

- [ ] **Step 5: Commit**

```bash
git add evaluators/shell_rft_v0/sandbox.py evaluators/shell_rft_v0/tests/test_sandbox.py
git commit -m "feat(evaluator): implement workspace materialization and sandboxed execution"
```

### Task 5: Evaluator entry point

**Files:**
- Create: `evaluators/shell_rft_v0/tests/test_evaluate.py`
- Modify: `evaluators/shell_rft_v0/main.py`

- [ ] **Step 1: Write the failing tests**

Create `evaluators/shell_rft_v0/tests/test_evaluate.py`:

```python
"""End-to-end tests for the evaluator reward function."""

from shell_rft_v0.main import evaluate


def _make_messages(assistant_content: str) -> list[dict]:
    """Helper: system + user + assistant messages."""
    return [
        {"role": "system", "content": "You solve toy shell tasks."},
        {"role": "user", "content": "Task: Count the .py files.\n\nFilesystem summary:\n- src/a.py\n- src/b.py\n- data/c.txt"},
        {"role": "assistant", "content": assistant_content},
    ]


def _workspace_spec():
    return {
        "files": [
            {"path": "src/a.py", "content": "# a\n"},
            {"path": "src/b.py", "content": "# b\n"},
            {"path": "data/c.txt", "content": "hello\n"},
        ]
    }


class TestEvaluateCorrect:
    def test_correct_find_wc(self):
        result = evaluate(
            messages=_make_messages("find . -name '*.py' | wc -l"),
            workspace_spec=_workspace_spec(),
            expected_stdout="2\n",
            task_type="file_counting",
        )
        assert result["score"] == 1.0

    def test_correct_with_trailing_whitespace_in_expected(self):
        result = evaluate(
            messages=_make_messages("find . -name '*.py' | wc -l"),
            workspace_spec=_workspace_spec(),
            expected_stdout="2\n\n",
            task_type="file_counting",
        )
        assert result["score"] == 1.0


class TestEvaluateIncorrect:
    def test_wrong_count(self):
        result = evaluate(
            messages=_make_messages("find . -name '*.txt' | wc -l"),
            workspace_spec=_workspace_spec(),
            expected_stdout="2\n",
            task_type="file_counting",
        )
        assert result["score"] == 0.0

    def test_explanation_instead_of_command(self):
        result = evaluate(
            messages=_make_messages("I think the answer is 2.\nThere are 2 .py files."),
            workspace_spec=_workspace_spec(),
            expected_stdout="2\n",
            task_type="file_counting",
        )
        # Multiple lines of explanation — extraction should reject this.
        assert result["score"] == 0.0

    def test_empty_assistant_message(self):
        result = evaluate(
            messages=_make_messages(""),
            workspace_spec=_workspace_spec(),
            expected_stdout="2\n",
            task_type="file_counting",
        )
        assert result["score"] == 0.0

    def test_policy_violation(self):
        result = evaluate(
            messages=_make_messages("rm -rf /"),
            workspace_spec=_workspace_spec(),
            expected_stdout="2\n",
            task_type="file_counting",
        )
        assert result["score"] == 0.0
        assert "reason" in result

    def test_multiple_commands_rejected(self):
        result = evaluate(
            messages=_make_messages("find . -name '*.py'\nwc -l"),
            workspace_spec=_workspace_spec(),
            expected_stdout="2\n",
            task_type="file_counting",
        )
        assert result["score"] == 0.0


class TestEvaluateEdgeCases:
    def test_command_in_markdown_code_block(self):
        text = "```bash\nfind . -name '*.py' | wc -l\n```"
        result = evaluate(
            messages=_make_messages(text),
            workspace_spec=_workspace_spec(),
            expected_stdout="2\n",
            task_type="file_counting",
        )
        assert result["score"] == 1.0

    def test_execution_failure(self):
        result = evaluate(
            messages=_make_messages("find --invalid-flag"),
            workspace_spec=_workspace_spec(),
            expected_stdout="2\n",
            task_type="file_counting",
        )
        assert result["score"] == 0.0

    def test_no_assistant_message(self):
        messages = [
            {"role": "system", "content": "You solve toy shell tasks."},
            {"role": "user", "content": "Count the .py files."},
        ]
        result = evaluate(
            messages=messages,
            workspace_spec=_workspace_spec(),
            expected_stdout="2\n",
            task_type="file_counting",
        )
        assert result["score"] == 0.0

    def test_missing_workspace_spec(self):
        result = evaluate(
            messages=_make_messages("find . -name '*.py' | wc -l"),
            workspace_spec=None,
            expected_stdout="2\n",
            task_type="file_counting",
        )
        assert result["score"] == 0.0

    def test_missing_expected_stdout(self):
        result = evaluate(
            messages=_make_messages("find . -name '*.py' | wc -l"),
            workspace_spec=_workspace_spec(),
            expected_stdout=None,
            task_type="file_counting",
        )
        assert result["score"] == 0.0

    def test_result_always_has_score_key(self):
        """Every result must have a 'score' key regardless of outcome."""
        cases = [
            _make_messages("find . -name '*.py' | wc -l"),
            _make_messages(""),
            _make_messages("rm -rf /"),
        ]
        for msgs in cases:
            result = evaluate(
                messages=msgs,
                workspace_spec=_workspace_spec(),
                expected_stdout="2\n",
                task_type="file_counting",
            )
            assert "score" in result
            assert isinstance(result["score"], float)

    def test_internal_error_returns_zero(self):
        """Evaluator must never crash — unexpected errors return score 0.0."""
        result = evaluate(
            messages=_make_messages("find . -type f | wc -l"),
            workspace_spec={"files": "not-a-list"},  # Malformed spec
            expected_stdout="2\n",
            task_type="file_counting",
        )
        assert result["score"] == 0.0
        assert "score" in result

    def test_wc_leading_whitespace_normalized(self):
        """wc -l may produce leading whitespace (e.g. '  2'); normalization
        strips trailing whitespace per line but leading whitespace is preserved
        in stdout. The expected_stdout should match what the command produces."""
        result = evaluate(
            messages=_make_messages("find . -name '*.py' | wc -l"),
            workspace_spec=_workspace_spec(),
            expected_stdout="2\n",
            task_type="file_counting",
        )
        # wc -l on Linux typically does not add leading whitespace when
        # reading from stdin (pipe), so this should match.
        assert result["score"] == 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd evaluators/shell_rft_v0 && uv run pytest tests/test_evaluate.py -v`
Expected: FAIL with `NotImplementedError`

- [ ] **Step 3: Implement the evaluator**

Replace contents of `evaluators/shell_rft_v0/main.py`:

```python
"""Evaluator entry point for Fireworks Build SDK."""

import shutil
import tempfile

from reward_kit import reward_function

from shell_rft_v0.normalize import normalize_stdout
from shell_rft_v0.policy import extract_single_command, validate_command
from shell_rft_v0.sandbox import materialize_workspace, run_command


@reward_function(id="shell-rft-v0")
def evaluate(
    messages: list[dict],
    workspace_spec: dict | None = None,
    expected_stdout: str | None = None,
    task_type: str | None = None,
    **kwargs,
) -> dict:
    """Score a single model response.

    Returns {"score": 1.0} on success, {"score": 0.0} otherwise.
    """
    try:
        return _evaluate_inner(messages, workspace_spec, expected_stdout, task_type)
    except Exception:
        return {"score": 0.0, "reason": "internal error"}


def _evaluate_inner(
    messages: list[dict],
    workspace_spec: dict | None,
    expected_stdout: str | None,
    task_type: str | None,
) -> dict:
    if workspace_spec is None or expected_stdout is None:
        return {"score": 0.0, "reason": "missing ground-truth fields"}

    # Find the last assistant message.
    assistant_content = ""
    for msg in reversed(messages):
        if msg.get("role") == "assistant":
            assistant_content = msg.get("content", "")
            break

    # Extract single command.
    command = extract_single_command(assistant_content)
    if command is None:
        return {"score": 0.0, "reason": "no single command extracted"}

    # Validate against policy.
    violation = validate_command(command)
    if violation is not None:
        return {"score": 0.0, "reason": violation}

    # Materialize workspace and execute.
    workdir = tempfile.mkdtemp(prefix="shell_rft_")
    try:
        materialize_workspace(workspace_spec, workdir)
        exit_code, stdout, stderr = run_command(command, workdir)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    if exit_code != 0:
        return {"score": 0.0, "reason": f"command exited with code {exit_code}"}

    actual = normalize_stdout(stdout, task_type=task_type)
    expected = normalize_stdout(expected_stdout, task_type=task_type)

    if actual == expected:
        return {"score": 1.0}

    return {"score": 0.0, "reason": "output mismatch"}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd evaluators/shell_rft_v0 && uv run pytest tests/test_evaluate.py -v`
Expected: All 15 tests PASS

- [ ] **Step 5: Commit**

```bash
git add evaluators/shell_rft_v0/main.py evaluators/shell_rft_v0/tests/test_evaluate.py
git commit -m "feat(evaluator): implement end-to-end reward function"
```

### Task 6: Final verification

- [ ] **Step 1: Run all evaluator tests**

Run: `cd evaluators/shell_rft_v0 && uv run pytest -v`
Expected: All tests PASS (2 smoke + 7 normalize + 31 policy + 10 sandbox + 15 evaluate = 65 tests)

- [ ] **Step 2: Run root project tests to check nothing is broken**

Run: `uv run pytest -v`
Expected: All 14 root tests PASS

- [ ] **Step 3: Generate a dataset and smoke-test the evaluator against it**

Run a quick manual check to verify the evaluator can score a known-correct command against a generated example:

```bash
cd evaluators/shell_rft_v0 && uv run python -c "
from shell_rft_v0.main import evaluate
result = evaluate(
    messages=[
        {'role': 'system', 'content': 'test'},
        {'role': 'user', 'content': 'Count py files'},
        {'role': 'assistant', 'content': \"find . -name '*.py' | wc -l\"},
    ],
    workspace_spec={'files': [
        {'path': 'a.py', 'content': 'x'},
        {'path': 'b.py', 'content': 'y'},
        {'path': 'c.txt', 'content': 'z'},
    ]},
    expected_stdout='2\n',
    task_type='file_counting',
)
print(result)
assert result['score'] == 1.0, f'Expected 1.0, got {result}'
print('Smoke test passed!')
"
```

- [ ] **Step 4: Update milestone status**

Update this file's status from `NOT STARTED` to `COMPLETE`.

---

## Deliverables

1. `evaluators/shell_rft_v0/normalize.py` — implemented
2. `evaluators/shell_rft_v0/policy.py` — `extract_single_command` and `validate_command` implemented
3. `evaluators/shell_rft_v0/sandbox.py` — `materialize_workspace` and `run_command` implemented
4. `evaluators/shell_rft_v0/main.py` — `evaluate` reward function wired up
5. `evaluators/shell_rft_v0/tests/test_normalize.py`
6. `evaluators/shell_rft_v0/tests/test_policy.py`
7. `evaluators/shell_rft_v0/tests/test_sandbox.py`
8. `evaluators/shell_rft_v0/tests/test_evaluate.py`
9. All existing + new tests pass
