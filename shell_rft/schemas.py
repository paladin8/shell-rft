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
class Example:
    messages: list[dict[str, str]]
    workspace_spec: WorkspaceSpec
    expected_stdout: str
    task_type: TaskType
