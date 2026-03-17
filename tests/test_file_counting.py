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


def test_expected_stdout_is_positive():
    """All file counts must be positive integers (no trivial zero-count examples)."""
    for ex in generate_file_counting_examples(100, random.Random(42)):
        count = int(ex.expected_stdout.strip())
        assert count > 0


def test_all_sub_types_exercised():
    """All 4 sub-types should appear across a reasonable sample."""
    examples = generate_file_counting_examples(100, random.Random(42))
    tasks = {ex.messages[1]["content"] for ex in examples}
    # Each sub-type uses a distinctive phrase in the task description.
    patterns = [
        "Count the .",       # _count_by_extension
        'starts with "test_"',  # _count_by_name_pattern
        "How many files are in",  # _count_all_in_subtree
        "contain",           # _count_files_containing
    ]
    for pattern in patterns:
        assert any(pattern in t for t in tasks), (
            f"No examples found matching sub-type pattern: {pattern!r}"
        )
