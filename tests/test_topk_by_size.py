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
