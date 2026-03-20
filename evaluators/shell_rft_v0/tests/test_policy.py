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


def test_extract_strips_think_blocks():
    text = "<think>\nLet me think.\n</think>\nfind . -type f | wc -l"
    assert extract_single_command(text) == "find . -type f | wc -l"


# --- validate_command: allowed commands ---


def test_validate_simple_allowed():
    assert validate_command("find . -type f") is None


def test_validate_pipeline_allowed():
    assert validate_command("find . -name '*.py' | wc -l") is None


def test_validate_multi_pipe_allowed():
    assert validate_command("find . -type f | sort | head -5") is None


def test_validate_allows_relative_paths():
    assert validate_command("find src/ -name '*.py'") is None


def test_validate_allows_grep_with_flags():
    assert validate_command("grep -rl 'ERROR' logs/") is None


def test_validate_allows_awk_program():
    assert validate_command("awk '{print $1}' data.csv") is None


def test_validate_allows_awk_with_comparison_operator():
    """Redirection check must not reject > inside quoted awk programs."""
    assert validate_command("awk -F',' '$2 > 80 {print $1}' data.csv") is None


# --- validate_command: quoted operators (the RFT-learned patterns) ---


def test_validate_allows_awk_logical_and():
    """&& inside awk single quotes is an awk operator, not shell."""
    assert validate_command("awk -F, 'NR>1 && $2 == \"down\" {print $1}' data.csv") is None


def test_validate_allows_awk_logical_or():
    """|| inside awk single quotes is an awk operator, not shell."""
    assert validate_command("awk 'NR==2 || NR==3 {print $2}' data.csv") is None


def test_validate_allows_awk_semicolon():
    """Semicolons inside awk quotes are awk statement separators."""
    assert validate_command("awk -F';' '{print $1}' data.csv") is None


def test_validate_allows_find_exec_escaped_semicolon():
    """find -exec {} \\; uses an escaped semicolon, not a shell operator."""
    assert validate_command("find . -exec cat {} \\;") is None


def test_validate_allows_find_exec_in_pipeline():
    assert validate_command("find . -exec grep -L 'x' {} \\; | sort") is None


# --- validate_command: disallowed commands ---


def test_validate_rejects_disallowed_command():
    reason = validate_command("rm -rf /")
    assert reason is not None
    assert "rm" in reason


def test_validate_rejects_disallowed_in_pipeline():
    reason = validate_command("find . | rm")
    assert reason is not None


# --- validate_command: shell operators ---


def test_validate_rejects_semicolon():
    reason = validate_command("find . -type f; rm -rf /")
    assert reason is not None


def test_validate_rejects_and_operator():
    reason = validate_command("find . && rm -rf /")
    assert reason is not None


def test_validate_rejects_or_operator():
    reason = validate_command("find . || echo fail")
    assert reason is not None


# --- validate_command: subshells ---


def test_validate_rejects_subshell():
    reason = validate_command("find . -exec $(rm) \\;")
    assert reason is not None


def test_validate_rejects_backtick_subshell():
    reason = validate_command("find . -name `echo test`")
    assert reason is not None


def test_validate_rejects_subshell_in_double_quotes():
    """$() inside double quotes IS expanded by the shell — must reject."""
    reason = validate_command('grep "$(cat /etc/passwd)" file.txt')
    assert reason is not None
    assert "subshell" in reason


def test_validate_rejects_backtick_in_double_quotes():
    """Backticks inside double quotes ARE expanded — must reject."""
    reason = validate_command('grep "`cat /etc/passwd`" file.txt')
    assert reason is not None


# --- validate_command: process substitution ---


def test_validate_rejects_process_substitution():
    reason = validate_command("sort <(find . -name '*.txt')")
    assert reason is not None
    assert "process substitution" in reason


# --- validate_command: redirections ---


def test_validate_rejects_redirect():
    reason = validate_command("find . > output.txt")
    assert reason is not None


def test_validate_rejects_append_redirect():
    reason = validate_command("find . >> output.txt")
    assert reason is not None


def test_validate_rejects_input_redirect():
    reason = validate_command("sort < input.txt")
    assert reason is not None


def test_validate_rejects_stderr_redirect():
    """2>/dev/null is still a redirect."""
    reason = validate_command("find . -type f 2>/dev/null")
    assert reason is not None


def test_validate_rejects_fd_redirect():
    reason = validate_command("find . 1>out.txt")
    assert reason is not None


# --- validate_command: xargs and find -exec with disallowed commands ---


def test_validate_rejects_disallowed_via_xargs():
    """xargs can run arbitrary commands — must validate the target command."""
    reason = validate_command("find . | xargs rm -f")
    assert reason is not None
    assert "rm" in reason


def test_validate_allows_xargs_with_allowed_command():
    assert validate_command("find . -name '*.txt' | xargs grep -l 'error'") is None


def test_validate_rejects_disallowed_via_find_exec():
    """find -exec can run arbitrary commands — must validate the target."""
    reason = validate_command("find . -exec rm {} \\;")
    assert reason is not None
    assert "rm" in reason


def test_validate_allows_find_exec_with_allowed_command():
    assert validate_command("find . -exec grep -l 'error' {} \\;") is None


def test_validate_rejects_disallowed_via_find_execdir():
    reason = validate_command("find . -execdir sh -c 'rm {}' \\;")
    assert reason is not None
    assert "sh" in reason


# --- validate_command: paths ---


def test_validate_rejects_absolute_path():
    reason = validate_command("cat /etc/passwd")
    assert reason is not None


def test_validate_rejects_path_traversal():
    reason = validate_command("cat ../../../etc/passwd")
    assert reason is not None


def test_validate_rejects_bare_dotdot():
    """Path traversal without a slash must still be caught."""
    reason = validate_command("cat ..")
    assert reason is not None


def test_validate_rejects_dotdot_in_relative_path():
    reason = validate_command("cat foo/../../../etc/passwd")
    assert reason is not None


def test_validate_rejects_tilde_expansion():
    reason = validate_command("find ~ -type f")
    assert reason is not None
    assert "~" in reason


def test_validate_rejects_tilde_user():
    reason = validate_command("cat ~root/.bashrc")
    assert reason is not None


# --- validate_command: compound commands ---


def test_validate_rejects_for_loop():
    reason = validate_command("for f in *.txt; do cat $f; done")
    assert reason is not None
    assert "compound" in reason


def test_validate_rejects_subshell_grouping():
    reason = validate_command("(find . -type f)")
    assert reason is not None


def test_validate_rejects_while_loop():
    reason = validate_command("while true; do find .; done")
    assert reason is not None


# --- validate_command: dangerous find flags ---


def test_validate_rejects_find_delete():
    reason = validate_command("find . -type f -delete")
    assert reason is not None
    assert "-delete" in reason


def test_validate_rejects_find_ok():
    reason = validate_command("find . -ok rm {} \\;")
    assert reason is not None
