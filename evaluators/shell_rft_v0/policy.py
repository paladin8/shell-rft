"""Command allowlist/blocklist validation."""

import re

import bashlex

ALLOWED_COMMANDS = frozenset([
    "find", "grep", "sort", "head", "tail",
    "wc", "cut", "awk", "sed", "cat", "xargs", "tr",
])

# Dangerous find flags that modify the filesystem or execute commands.
_DANGEROUS_FIND_FLAGS = frozenset([
    "-delete", "-ok", "-okdir", "-fprint", "-fls", "-fprintf",
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


def _walk_nodes(node):
    """Yield all nodes in a bashlex AST."""
    yield node
    for child in getattr(node, "parts", None) or []:
        yield from _walk_nodes(child)
    # Some nodes store children in 'list' (e.g. ListNode).
    for child in getattr(node, "list", None) or []:
        yield from _walk_nodes(child)


def validate_command(command: str) -> str | None:
    """Check command against policy. Returns violation reason or None if OK."""
    try:
        parts = bashlex.parse(command)
    except bashlex.errors.ParsingError:
        return "malformed command"

    # Must be a single command or pipeline — not a list (&&, ||, ;).
    if len(parts) != 1:
        return "multiple statements are not allowed"

    root = parts[0]

    # Reject compound commands (&&, ||, ;).
    if root.kind == "list":
        for child in _walk_nodes(root):
            if child.kind == "operator" and child.op in ("&&", "||", ";"):
                return f"shell operator '{child.op}' is not allowed"
        return "compound commands are not allowed"

    # Reject compound shell constructs (for, while, if, subshell groups, braces).
    if root.kind == "compound":
        return "compound commands (loops, conditionals, subshells) are not allowed"

    # Walk entire AST looking for disallowed constructs.
    for node in _walk_nodes(root):
        # Reject command substitution $() and backticks.
        if node.kind == "commandsubstitution":
            return "subshells are not allowed"

        # Reject process substitution <() and >().
        if node.kind == "processsubstitution":
            return "process substitution is not allowed"

        # Reject redirections.
        if node.kind == "redirect":
            return "redirections are not allowed"

    # Collect all commands in the pipeline and validate each.
    command_nodes = [
        node for node in _walk_nodes(root)
        if node.kind == "command"
    ]

    if not command_nodes:
        return "no command found"

    for cmd_node in command_nodes:
        words = [
            child.word for child in (cmd_node.parts or [])
            if child.kind == "word"
        ]
        if not words:
            continue

        cmd_name = words[0]
        if cmd_name not in ALLOWED_COMMANDS:
            return f"command '{cmd_name}' is not allowed"

        # Check for paths outside the workspace.
        for word in words[1:]:
            if word.startswith("/"):
                return f"absolute path '{word}' is not allowed"
            if word.startswith("~"):
                return f"path '{word}' outside workspace is not allowed"
            if ".." in word.split("/"):
                return f"path traversal in '{word}' is not allowed"

        # Validate commands invoked via xargs.
        if cmd_name == "xargs":
            # Find the first non-flag argument (the command xargs will run).
            for word in words[1:]:
                if not word.startswith("-"):
                    if word not in ALLOWED_COMMANDS:
                        return f"command '{word}' via xargs is not allowed"
                    break

        # Validate find-specific concerns.
        if cmd_name == "find":
            for j, word in enumerate(words):
                # Reject dangerous find flags.
                if word in _DANGEROUS_FIND_FLAGS:
                    return f"find flag '{word}' is not allowed"
                # Validate commands invoked via -exec / -execdir / -ok / -okdir.
                if word in ("-exec", "-execdir", "-ok", "-okdir") and j + 1 < len(words):
                    exec_cmd = words[j + 1]
                    if exec_cmd not in ALLOWED_COMMANDS:
                        return f"command '{exec_cmd}' via {word} is not allowed"

    return None
