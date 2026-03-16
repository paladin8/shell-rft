"""Smoke tests to verify project structure and imports."""

from shell_rft.prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE
from shell_rft.schemas import Example, FileSpec, WorkspaceSpec


def test_system_prompt_is_nonempty():
    assert len(SYSTEM_PROMPT) > 0


def test_user_prompt_template_has_placeholders():
    assert "{task}" in USER_PROMPT_TEMPLATE
    assert "{filesystem_summary}" in USER_PROMPT_TEMPLATE


def test_example_round_trips():
    ex = Example(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "Count .txt files"},
        ],
        workspace_spec=WorkspaceSpec(files=[
            FileSpec(path="a.txt", content="hello"),
        ]),
        expected_stdout="1\n",
        task_type="file_counting",
    )
    assert ex.task_type == "file_counting"
    assert len(ex.workspace_spec.files) == 1
