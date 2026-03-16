"""Workspace materialization and sandboxed command execution."""


def materialize_workspace(workspace_spec: dict, workdir: str) -> None:
    """Write workspace_spec files into workdir."""
    raise NotImplementedError


def run_command(command: str, cwd: str, timeout_s: float = 2.0) -> tuple[int, str, str]:
    """Run a shell command in cwd. Returns (exit_code, stdout, stderr)."""
    raise NotImplementedError
