"""Eval-protocol evaluation test wrapper for shell-rft evaluator.

This wraps the existing @reward_function evaluator in an @evaluation_test
so that eval-protocol can discover, validate, and deploy it.

Run from evaluators/shell_rft_v0/:
    uv run eval-protocol create rft --base-model accounts/fireworks/models/qwen3-8b ...
"""

import shutil
import tempfile
from typing import Any

from eval_protocol.models import EvaluateResult, EvaluationRow, InputMetadata, Message
from eval_protocol.pytest import NoOpRolloutProcessor
from eval_protocol.pytest.evaluation_test import evaluation_test

from shell_rft_v0.normalize import normalize_stdout
from shell_rft_v0.policy import extract_single_command, validate_command
from shell_rft_v0.sandbox import materialize_workspace, run_command


def _dataset_adapter(rows: list[dict[str, Any]]) -> list[EvaluationRow]:
    """Convert shell-rft JSONL rows to EvaluationRow objects.

    Stashes workspace_spec, expected_stdout, and task_type in
    input_metadata.dataset_info so the test function can access them.
    """
    result = []
    for i, row in enumerate(rows):
        messages = [Message(**m) for m in row["messages"]]
        eval_row = EvaluationRow(
            messages=messages,
            ground_truth=row.get("expected_stdout", ""),
            input_metadata=InputMetadata(
                row_id=str(i),
                dataset_info={
                    "workspace_spec": row["workspace_spec"],
                    "expected_stdout": row["expected_stdout"],
                    "task_type": row.get("task_type", "unknown"),
                },
            ),
        )
        result.append(eval_row)
    return result


def _score_response(
    assistant_content: str,
    workspace_spec: dict,
    expected_stdout: str,
    task_type: str,
) -> EvaluateResult:
    """Run the evaluator pipeline on a model response."""
    command = extract_single_command(assistant_content)
    if command is None:
        return EvaluateResult(
            score=0.0, reason="no single command extracted", metrics={}
        )

    violation = validate_command(command)
    if violation is not None:
        return EvaluateResult(score=0.0, reason=violation, metrics={})

    workdir = tempfile.mkdtemp(prefix="shell_rft_ep_")
    try:
        materialize_workspace(workspace_spec, workdir)
        exit_code, stdout, stderr = run_command(command, workdir)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    if exit_code != 0:
        return EvaluateResult(
            score=0.0, reason=f"command exited with code {exit_code}", metrics={}
        )

    actual = normalize_stdout(stdout, task_type=task_type)
    expected = normalize_stdout(expected_stdout, task_type=task_type)

    if actual == expected:
        return EvaluateResult(score=1.0, metrics={})

    return EvaluateResult(score=0.0, reason="output mismatch", metrics={})


@evaluation_test(
    input_dataset=["../../../data/train.jsonl"],
    dataset_adapter=_dataset_adapter,
    rollout_processor=NoOpRolloutProcessor(),
    passed_threshold=0.5,
    max_dataset_rows=5,
    mode="pointwise",
)
def test_shell_rft_eval(row: EvaluationRow) -> EvaluationRow:
    """Evaluate a shell command response using the shell-rft evaluator."""
    # Find the last assistant message (the model's rollout response).
    assistant_content = ""
    for msg in reversed(row.messages):
        if msg.role == "assistant":
            content = msg.content
            if isinstance(content, list):
                assistant_content = "".join(
                    getattr(p, "text", str(p)) for p in content
                )
            else:
                assistant_content = str(content or "")
            break

    # Get ground-truth fields from dataset_info.
    info = row.input_metadata.dataset_info or {}
    workspace_spec = info.get("workspace_spec", {})
    expected_stdout = info.get("expected_stdout", "")
    task_type = info.get("task_type", "unknown")

    row.evaluation_result = _score_response(
        assistant_content, workspace_spec, expected_stdout, task_type
    )
    return row
