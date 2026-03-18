"""Content-search task family generator."""

from __future__ import annotations

import random

from shell_rft.prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE
from shell_rft.schemas import Example, FileSpec, WorkspaceSpec

_DIRS = ["src", "logs", "data", "app", "lib", "tests", "docs", "config"]
_SUBDIRS = ["utils", "core", "api", "auth", "models"]
_EXTENSIONS = [".py", ".txt", ".log", ".csv", ".json"]
_STEMS = [
    "main", "utils", "config", "app", "debug", "error",
    "output", "report", "index", "helper",
]
_SEARCH_TERMS = ["ERROR", "TODO", "import", "timeout", "hello", "DEBUG"]
_CONTENT_POOL = [
    "# placeholder",
    "import os",
    "ERROR: connection refused",
    "DEBUG: starting service",
    "name,value",
    "hello world",
    "TODO: fix this",
    "print('hello')",
    "status=ok",
    "timeout exceeded",
    "WARNING: disk space low",
    "import sys",
    "return None",
    "DEBUG: request received",
    "ERROR: timeout exceeded",
]


def _build_workspace(rng: random.Random) -> list[FileSpec]:
    """Create a random set of files across a small directory tree."""
    n_dirs = rng.randint(2, 5)
    n_files = rng.randint(5, 20)
    dirs = rng.sample(_DIRS, n_dirs)

    files: list[FileSpec] = []
    seen: set[str] = set()

    for i in range(n_files):
        d = rng.choice(dirs)
        if rng.random() < 0.3:
            d = f"{d}/{rng.choice(_SUBDIRS)}"
        stem = rng.choice(_STEMS)
        ext = rng.choice(_EXTENSIONS)
        path = f"{d}/{stem}{ext}"

        if path in seen:
            path = f"{d}/{stem}_{i}{ext}"
        if path in seen:
            continue
        seen.add(path)

        n_lines = rng.randint(1, 5)
        content = "\n".join(rng.choices(_CONTENT_POOL, k=n_lines)) + "\n"
        files.append(FileSpec(path=path, content=content))

    return files


def _filesystem_summary(files: list[FileSpec]) -> str:
    return "\n".join(f"- {f.path}" for f in files)


def _make_example(task: str, files: list[FileSpec], expected: str) -> Example:
    user_content = USER_PROMPT_TEMPLATE.format(
        task=task,
        filesystem_summary=_filesystem_summary(files),
    )
    return Example(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        workspace_spec=WorkspaceSpec(files=files),
        expected_stdout=expected,
        task_type="content_search",
    )


def _find_files_containing(rng: random.Random) -> Example:
    """List files containing a search term under a directory, sorted."""
    while True:
        files = _build_workspace(rng)
        dirs = sorted({f.path.split("/")[0] for f in files})
        target_dir = rng.choice(dirs)
        terms = list(_SEARCH_TERMS)
        rng.shuffle(terms)
        for term in terms:
            matches = sorted(
                f.path for f in files
                if f.path.startswith(f"{target_dir}/") and term in f.content
            )
            if matches:
                task = (
                    f"Print the paths of files under {target_dir}/ that "
                    f"contain '{term}', one per line, sorted."
                )
                expected = "\n".join(matches) + "\n"
                return _make_example(task, files, expected)


def _grep_lines_from_file(rng: random.Random) -> Example:
    """Print matching lines from a specific file."""
    while True:
        files = _build_workspace(rng)
        candidates = list(files)
        rng.shuffle(candidates)
        for f in candidates:
            lines = [line for line in f.content.split("\n") if line]
            terms = list(_SEARCH_TERMS)
            rng.shuffle(terms)
            for term in terms:
                matching = [line for line in lines if term in line]
                if matching:
                    task = (
                        f"Print the lines from {f.path} that contain '{term}'."
                    )
                    expected = "\n".join(matching) + "\n"
                    return _make_example(task, files, expected)


def _find_files_not_containing(rng: random.Random) -> Example:
    """List files NOT containing a search term under a directory, sorted."""
    while True:
        files = _build_workspace(rng)
        dirs = sorted({f.path.split("/")[0] for f in files})
        target_dir = rng.choice(dirs)
        dir_files = [f for f in files if f.path.startswith(f"{target_dir}/")]
        if len(dir_files) < 2:
            continue
        terms = list(_SEARCH_TERMS)
        rng.shuffle(terms)
        for term in terms:
            containing = [f for f in dir_files if term in f.content]
            not_containing = sorted(
                f.path for f in dir_files if term not in f.content
            )
            # Need both some containing and some not for a useful task.
            if containing and not_containing:
                task = (
                    f"Print the paths of files under {target_dir}/ that do NOT "
                    f"contain '{term}', one per line, sorted."
                )
                expected = "\n".join(not_containing) + "\n"
                return _make_example(task, files, expected)


_SUB_TYPES = [
    _find_files_containing,
    _grep_lines_from_file,
    _find_files_not_containing,
]


def generate_content_search_examples(
    n: int,
    rng: random.Random,
) -> list[Example]:
    """Generate n content-search examples using the given RNG."""
    return [rng.choice(_SUB_TYPES)(rng) for _ in range(n)]
