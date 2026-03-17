"""Workspace materialization and sandboxed command execution."""

import os
import subprocess


def materialize_workspace(workspace_spec: dict, workdir: str) -> None:
    """Write workspace_spec files into workdir."""
    real_workdir = os.path.realpath(workdir)
    for file_entry in workspace_spec.get("files", []):
        file_path = os.path.realpath(os.path.join(workdir, file_entry["path"]))
        if not file_path.startswith(real_workdir + os.sep) and file_path != real_workdir:
            raise ValueError(f"workspace path escapes workdir: {file_entry['path']}")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as f:
            f.write(file_entry["content"])


def run_command(
    command: str, cwd: str, timeout_s: float = 2.0
) -> tuple[int, str, str]:
    """Run a shell command in cwd. Returns (exit_code, stdout, stderr)."""
    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        return (result.returncode, result.stdout, result.stderr)
    except subprocess.TimeoutExpired:
        return (124, "", "timeout")
