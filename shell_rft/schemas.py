from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


TaskType = Literal["file_counting", "content_search", "topk_by_size", "csv_filtering"]


@dataclass
class FileSpec:
    path: str
    content: str


@dataclass
class WorkspaceSpec:
    files: list[FileSpec] = field(default_factory=list)


@dataclass
class GroundTruth:
    """Ground-truth fields bundled for eval-protocol's EvaluationRow.ground_truth."""
    workspace_spec: WorkspaceSpec
    expected_stdout: str
    task_type: str


@dataclass
class Example:
    messages: list[dict[str, str]]
    workspace_spec: WorkspaceSpec
    expected_stdout: str
    task_type: TaskType
    ground_truth: GroundTruth | None = None

    def __post_init__(self) -> None:
        if self.ground_truth is None:
            self.ground_truth = GroundTruth(
                workspace_spec=self.workspace_spec,
                expected_stdout=self.expected_stdout,
                task_type=self.task_type,
            )
