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
