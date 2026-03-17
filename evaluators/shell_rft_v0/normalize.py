"""Stdout normalization for comparison."""


def normalize_stdout(stdout: str, task_type: str | None = None) -> str:
    """Normalize stdout for comparison.

    Strips trailing whitespace from each line, collapses trailing newlines,
    and ensures exactly one trailing newline.
    """
    lines = stdout.split("\n")
    # Remove trailing empty strings from split (trailing newlines)
    while lines and lines[-1] == "":
        lines.pop()
    # Strip trailing whitespace from each line
    lines = [line.rstrip() for line in lines]
    # Re-join with single trailing newline
    return "\n".join(lines) + "\n"
