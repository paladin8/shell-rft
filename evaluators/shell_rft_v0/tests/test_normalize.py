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
