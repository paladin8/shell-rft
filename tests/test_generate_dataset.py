"""Integration tests for the dataset generation script."""

import json
import subprocess
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_script_produces_jsonl_files(tmp_path):
    result = subprocess.run(
        [
            "uv", "run", "python", "scripts/generate_dataset.py",
            "--train-n", "5",
            "--val-n", "3",
            "--test-n", "2",
            "--output-dir", str(tmp_path),
        ],
        capture_output=True,
        text=True,
        cwd=_PROJECT_ROOT,
    )
    assert result.returncode == 0, result.stderr

    for name, expected_n in [("train", 5), ("val", 3), ("test", 2)]:
        path = tmp_path / f"{name}.jsonl"
        assert path.exists(), f"{name}.jsonl not found"
        lines = path.read_text().strip().splitlines()
        assert len(lines) == expected_n, (
            f"{name}.jsonl has {len(lines)} lines, expected {expected_n}"
        )


def test_jsonl_rows_are_valid(tmp_path):
    subprocess.run(
        [
            "uv", "run", "python", "scripts/generate_dataset.py",
            "--train-n", "3",
            "--val-n", "1",
            "--test-n", "1",
            "--output-dir", str(tmp_path),
        ],
        capture_output=True,
        text=True,
        check=True,
        cwd=_PROJECT_ROOT,
    )
    for jsonl_file in tmp_path.glob("*.jsonl"):
        for line in jsonl_file.read_text().strip().splitlines():
            row = json.loads(line)
            assert "messages" in row
            assert "workspace_spec" in row
            assert "expected_stdout" in row
            assert "task_type" in row
