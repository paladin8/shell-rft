"""End-to-end tests for the evaluator reward function."""

from shell_rft_v0.main import evaluate


def _make_messages(assistant_content: str) -> list[dict]:
    """Helper: system + user + assistant messages."""
    return [
        {"role": "system", "content": "You solve toy shell tasks."},
        {"role": "user", "content": "Task: Count the .py files.\n\nFilesystem summary:\n- src/a.py\n- src/b.py\n- data/c.txt"},
        {"role": "assistant", "content": assistant_content},
    ]


def _workspace_spec():
    return {
        "files": [
            {"path": "src/a.py", "content": "# a\n"},
            {"path": "src/b.py", "content": "# b\n"},
            {"path": "data/c.txt", "content": "hello\n"},
        ]
    }


class TestEvaluateCorrect:
    def test_correct_find_wc(self):
        result = evaluate(
            messages=_make_messages("find . -name '*.py' | wc -l"),
            workspace_spec=_workspace_spec(),
            expected_stdout="2\n",
            task_type="file_counting",
        )
        assert result["score"] == 1.0

    def test_correct_with_trailing_whitespace_in_expected(self):
        result = evaluate(
            messages=_make_messages("find . -name '*.py' | wc -l"),
            workspace_spec=_workspace_spec(),
            expected_stdout="2\n\n",
            task_type="file_counting",
        )
        assert result["score"] == 1.0


class TestEvaluateIncorrect:
    def test_wrong_count(self):
        result = evaluate(
            messages=_make_messages("find . -name '*.txt' | wc -l"),
            workspace_spec=_workspace_spec(),
            expected_stdout="2\n",
            task_type="file_counting",
        )
        assert result["score"] == 0.0

    def test_explanation_instead_of_command(self):
        result = evaluate(
            messages=_make_messages("I think the answer is 2.\nThere are 2 .py files."),
            workspace_spec=_workspace_spec(),
            expected_stdout="2\n",
            task_type="file_counting",
        )
        # Multiple lines of explanation — extraction should reject this.
        assert result["score"] == 0.0

    def test_empty_assistant_message(self):
        result = evaluate(
            messages=_make_messages(""),
            workspace_spec=_workspace_spec(),
            expected_stdout="2\n",
            task_type="file_counting",
        )
        assert result["score"] == 0.0

    def test_policy_violation(self):
        result = evaluate(
            messages=_make_messages("rm -rf /"),
            workspace_spec=_workspace_spec(),
            expected_stdout="2\n",
            task_type="file_counting",
        )
        assert result["score"] == 0.0
        assert "reason" in result

    def test_multiple_commands_rejected(self):
        result = evaluate(
            messages=_make_messages("find . -name '*.py'\nwc -l"),
            workspace_spec=_workspace_spec(),
            expected_stdout="2\n",
            task_type="file_counting",
        )
        assert result["score"] == 0.0


class TestEvaluateEdgeCases:
    def test_command_in_markdown_code_block(self):
        text = "```bash\nfind . -name '*.py' | wc -l\n```"
        result = evaluate(
            messages=_make_messages(text),
            workspace_spec=_workspace_spec(),
            expected_stdout="2\n",
            task_type="file_counting",
        )
        assert result["score"] == 1.0

    def test_execution_failure(self):
        result = evaluate(
            messages=_make_messages("find --invalid-flag"),
            workspace_spec=_workspace_spec(),
            expected_stdout="2\n",
            task_type="file_counting",
        )
        assert result["score"] == 0.0

    def test_no_assistant_message(self):
        messages = [
            {"role": "system", "content": "You solve toy shell tasks."},
            {"role": "user", "content": "Count the .py files."},
        ]
        result = evaluate(
            messages=messages,
            workspace_spec=_workspace_spec(),
            expected_stdout="2\n",
            task_type="file_counting",
        )
        assert result["score"] == 0.0

    def test_missing_workspace_spec(self):
        result = evaluate(
            messages=_make_messages("find . -name '*.py' | wc -l"),
            workspace_spec=None,
            expected_stdout="2\n",
            task_type="file_counting",
        )
        assert result["score"] == 0.0

    def test_missing_expected_stdout(self):
        result = evaluate(
            messages=_make_messages("find . -name '*.py' | wc -l"),
            workspace_spec=_workspace_spec(),
            expected_stdout=None,
            task_type="file_counting",
        )
        assert result["score"] == 0.0

    def test_result_always_has_score_key(self):
        """Every result must have a 'score' key regardless of outcome."""
        cases = [
            _make_messages("find . -name '*.py' | wc -l"),
            _make_messages(""),
            _make_messages("rm -rf /"),
        ]
        for msgs in cases:
            result = evaluate(
                messages=msgs,
                workspace_spec=_workspace_spec(),
                expected_stdout="2\n",
                task_type="file_counting",
            )
            assert "score" in result
            assert isinstance(result["score"], float)

    def test_internal_error_returns_zero(self):
        """Evaluator must never crash — unexpected errors return score 0.0."""
        result = evaluate(
            messages=_make_messages("find . -type f | wc -l"),
            workspace_spec={"files": "not-a-list"},  # Malformed spec
            expected_stdout="2\n",
            task_type="file_counting",
        )
        assert result["score"] == 0.0
        assert "score" in result

    def test_wc_leading_whitespace_normalized(self):
        """wc -l may produce leading whitespace (e.g. '  2'); normalization
        strips trailing whitespace per line but leading whitespace is preserved
        in stdout. The expected_stdout should match what the command produces."""
        result = evaluate(
            messages=_make_messages("find . -name '*.py' | wc -l"),
            workspace_spec=_workspace_spec(),
            expected_stdout="2\n",
            task_type="file_counting",
        )
        # wc -l on Linux typically does not add leading whitespace when
        # reading from stdin (pipe), so this should match.
        assert result["score"] == 1.0
