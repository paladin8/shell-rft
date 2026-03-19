"""Command allowlist/blocklist validation."""

import re
import shlex

ALLOWED_COMMANDS = frozenset([
    "find", "grep", "sort", "head", "tail",
    "wc", "cut", "awk", "sed", "cat", "xargs", "tr",
])


def extract_single_command(text: str) -> str | None:
    """Extract a single shell command from model output. Returns None if not exactly one."""
    # Strip Qwen3-style reasoning blocks.
    text = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)

    # Try to extract from a markdown fenced code block first.
    fence_match = re.search(r"```(?:\w*)\n(.*?)```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1)

    # Split into non-empty lines.
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]

    if len(lines) != 1:
        return None

    return lines[0]


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
