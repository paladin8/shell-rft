"""Evaluator entry point for Fireworks Build SDK."""

from reward_kit import reward_function


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
    raise NotImplementedError
