"""Smoke tests for the evaluator package."""

from shell_rft_v0.policy import ALLOWED_COMMANDS


def test_allowed_commands_matches_spec():
    expected = {
        "find", "grep", "sort", "head", "tail",
        "wc", "cut", "awk", "sed", "cat", "xargs", "tr",
        "ls", "du", "stat",
    }
    assert ALLOWED_COMMANDS == expected


def test_allowed_commands_is_frozen():
    assert isinstance(ALLOWED_COMMANDS, frozenset)
