"""Command allowlist/blocklist validation."""

ALLOWED_COMMANDS = frozenset([
    "find", "grep", "sort", "head", "tail",
    "wc", "cut", "awk", "sed", "cat", "xargs", "tr",
])


def validate_command(command: str) -> str | None:
    """Check command against policy. Returns violation reason or None if OK."""
    raise NotImplementedError


def extract_single_command(text: str) -> str | None:
    """Extract a single shell command from model output. Returns None if not exactly one."""
    raise NotImplementedError
