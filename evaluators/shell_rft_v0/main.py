"""Evaluator entry point for Fireworks Build SDK."""

import logging
import shutil
import tempfile

from reward_kit import reward_function

logger = logging.getLogger(__name__)

from shell_rft_v0.normalize import normalize_stdout
from shell_rft_v0.policy import extract_single_command, validate_command
from shell_rft_v0.sandbox import materialize_workspace, run_command


@reward_function(id="shell-rft-v0")
def evaluate(
    messages: list[dict],
    workspace_spec: dict | None = None,
    expected_stdout: str | None = None,
    task_type: str | None = None,
    **kwargs,
) -> dict:
    """Score a single model response.

    Returns {"score": 1.0} on success, {"score": 0.0} otherwise.
    """
    try:
        return _evaluate_inner(messages, workspace_spec, expected_stdout, task_type)
    except Exception:
        logger.exception("Evaluator internal error")
        return {"score": 0.0, "reason": "internal error"}


def _evaluate_inner(
    messages: list[dict],
    workspace_spec: dict | None,
    expected_stdout: str | None,
    task_type: str | None,
) -> dict:
    if workspace_spec is None or expected_stdout is None:
        return {"score": 0.0, "reason": "missing ground-truth fields"}

    # Find the last assistant message.
    assistant_content = ""
    for msg in reversed(messages):
        if msg.get("role") == "assistant":
            assistant_content = msg.get("content", "")
            break

    # Extract single command.
    command = extract_single_command(assistant_content)
    if command is None:
        return {"score": 0.0, "reason": "no single command extracted"}

    # Validate against policy.
    violation = validate_command(command)
    if violation is not None:
        return {"score": 0.0, "reason": violation}

    # Materialize workspace and execute.
    workdir = tempfile.mkdtemp(prefix="shell_rft_")
    try:
        materialize_workspace(workspace_spec, workdir)
        exit_code, stdout, stderr = run_command(command, workdir)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    if exit_code != 0:
        return {"score": 0.0, "reason": f"command exited with code {exit_code}"}

    actual = normalize_stdout(stdout, task_type=task_type)
    expected = normalize_stdout(expected_stdout, task_type=task_type)

    if actual == expected:
        return {"score": 1.0}

    return {"score": 0.0, "reason": "output mismatch"}
