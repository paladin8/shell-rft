"""Tests for workspace materialization and sandboxed execution."""

import os
import tempfile

import pytest

from shell_rft_v0.sandbox import materialize_workspace, run_command


class TestMaterializeWorkspace:
    def test_creates_files(self):
        spec = {
            "files": [
                {"path": "src/main.py", "content": "print('hello')\n"},
                {"path": "data/test.csv", "content": "a,b\n1,2\n"},
            ]
        }
        with tempfile.TemporaryDirectory() as workdir:
            materialize_workspace(spec, workdir)
            main_path = os.path.join(workdir, "src", "main.py")
            assert os.path.exists(main_path)
            with open(main_path) as f:
                assert f.read() == "print('hello')\n"

            csv_path = os.path.join(workdir, "data", "test.csv")
            assert os.path.exists(csv_path)
            with open(csv_path) as f:
                assert f.read() == "a,b\n1,2\n"

    def test_creates_nested_directories(self):
        spec = {
            "files": [
                {"path": "a/b/c/deep.txt", "content": "deep\n"},
            ]
        }
        with tempfile.TemporaryDirectory() as workdir:
            materialize_workspace(spec, workdir)
            assert os.path.exists(os.path.join(workdir, "a", "b", "c", "deep.txt"))

    def test_creates_root_level_file(self):
        spec = {
            "files": [
                {"path": "readme.txt", "content": "hello\n"},
            ]
        }
        with tempfile.TemporaryDirectory() as workdir:
            materialize_workspace(spec, workdir)
            path = os.path.join(workdir, "readme.txt")
            assert os.path.exists(path)
            with open(path) as f:
                assert f.read() == "hello\n"

    def test_rejects_path_traversal_in_spec(self):
        spec = {"files": [{"path": "../../escape.txt", "content": "bad\n"}]}
        with tempfile.TemporaryDirectory() as workdir:
            with pytest.raises(ValueError, match="escapes workdir"):
                materialize_workspace(spec, workdir)

    def test_empty_workspace(self):
        spec = {"files": []}
        with tempfile.TemporaryDirectory() as workdir:
            materialize_workspace(spec, workdir)
            # Should not error; workdir exists but is empty.
            assert os.path.isdir(workdir)


class TestRunCommand:
    def test_simple_echo(self):
        with tempfile.TemporaryDirectory() as workdir:
            exit_code, stdout, stderr = run_command("echo hello", workdir)
            assert exit_code == 0
            assert stdout == "hello\n"

    def test_command_respects_cwd(self):
        with tempfile.TemporaryDirectory() as workdir:
            # Create a file and find it.
            with open(os.path.join(workdir, "test.txt"), "w") as f:
                f.write("content\n")
            exit_code, stdout, _ = run_command("find . -name 'test.txt'", workdir)
            assert exit_code == 0
            assert "test.txt" in stdout

    def test_failed_command_returns_nonzero(self):
        with tempfile.TemporaryDirectory() as workdir:
            exit_code, _, _ = run_command("false", workdir)
            assert exit_code != 0

    def test_timeout_returns_124(self):
        with tempfile.TemporaryDirectory() as workdir:
            exit_code, _, _ = run_command("sleep 10", workdir, timeout_s=0.5)
            assert exit_code == 124

    def test_pipeline(self):
        with tempfile.TemporaryDirectory() as workdir:
            for name in ["a.py", "b.py", "c.txt"]:
                with open(os.path.join(workdir, name), "w") as f:
                    f.write("x\n")
            exit_code, stdout, _ = run_command(
                "find . -name '*.py' | wc -l", workdir
            )
            assert exit_code == 0
            assert stdout.strip() == "2"

    def test_stderr_captured(self):
        with tempfile.TemporaryDirectory() as workdir:
            exit_code, stdout, stderr = run_command(
                "echo err >&2", workdir
            )
            assert exit_code == 0
            assert stderr.strip() == "err"
