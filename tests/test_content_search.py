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
