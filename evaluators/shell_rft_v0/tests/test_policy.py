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


def test_validate_rejects_dotdot_grep_pattern():
    """Known limitation: grep '..' is rejected because .. looks like path traversal
    after shlex strips quotes. Acceptable false negative for v0's conservative policy."""
    reason = validate_command("grep '..' file.txt")
    assert reason is not None
