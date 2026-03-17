"""File-counting task family generator."""

from __future__ import annotations

import random

from shell_rft.prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE
from shell_rft.schemas import Example, FileSpec, WorkspaceSpec

_DIRS = ["src", "logs", "data", "app", "lib", "tests", "docs", "config"]
_SUBDIRS = ["utils", "config", "core", "api", "auth", "models"]
_EXTENSIONS = [".py", ".txt", ".log", ".csv", ".json"]
_STEMS = [
    "main", "utils", "config", "app", "debug", "error",
    "output", "report", "index", "helper",
]
_CONTENT_LINES = [
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
        # ~30% chance of a nested subdirectory
        if rng.random() < 0.3:
            d = f"{d}/{rng.choice(_SUBDIRS)}"
        stem = rng.choice(_STEMS)
        ext = rng.choice(_EXTENSIONS)
        prefix = "test_" if rng.random() < 0.25 else ""
        path = f"{d}/{prefix}{stem}{ext}"

        if path in seen:
            path = f"{d}/{prefix}{stem}_{i}{ext}"
        if path in seen:
            continue
        seen.add(path)

        n_lines = rng.randint(1, 5)
        content = "\n".join(rng.choices(_CONTENT_LINES, k=n_lines)) + "\n"
        files.append(FileSpec(path=path, content=content))

    return files


def _file_ext(path: str) -> str:
    """Return the file extension including the dot, e.g. '.py'."""
    dot = path.rfind(".")
    return path[dot:] if dot != -1 else ""


def _filesystem_summary(files: list[FileSpec]) -> str:
    return "\n".join(f"- {f.path}" for f in files)


def _make_example(task: str, files: list[FileSpec], count: int) -> Example:
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
        expected_stdout=f"{count}\n",
        task_type="file_counting",
    )


def _count_by_extension(rng: random.Random) -> Example:
    files = _build_workspace(rng)
    # Pick an extension that exists in the workspace to avoid zero counts.
    exts_present = sorted({_file_ext(f.path) for f in files})
    ext = rng.choice(exts_present)
    dirs = sorted({f.path.split("/")[0] for f in files})
    target_dir = rng.choice(dirs)
    count = sum(
        1 for f in files
        if f.path.startswith(f"{target_dir}/") and f.path.endswith(ext)
    )
    # If the chosen dir has none of the chosen ext, search all dirs instead.
    if count == 0:
        count = sum(1 for f in files if f.path.endswith(ext))
        task = f"Count the {ext} files."
    else:
        task = f"Count the {ext} files under {target_dir}/."
    return _make_example(task, files, count)


def _count_by_name_pattern(rng: random.Random) -> Example:
    # Rebuild workspace until at least one test_ file exists.
    while True:
        files = _build_workspace(rng)
        count = sum(
            1 for f in files
            if f.path.split("/")[-1].startswith("test_")
        )
        if count > 0:
            break
    task = 'Count files whose name starts with "test_".'
    return _make_example(task, files, count)


def _count_all_in_subtree(rng: random.Random) -> Example:
    files = _build_workspace(rng)
    dirs = sorted({f.path.split("/")[0] for f in files})
    target_dir = rng.choice(dirs)
    count = sum(1 for f in files if f.path.startswith(f"{target_dir}/"))
    task = f"How many files are in {target_dir}/?"
    return _make_example(task, files, count)


def _count_files_containing(rng: random.Random) -> Example:
    files = _build_workspace(rng)
    search_terms = ["ERROR", "TODO", "import", "hello", "status"]
    rng.shuffle(search_terms)
    # Pick the first search term that matches at least one file.
    search_term = search_terms[0]
    count = 0
    for term in search_terms:
        count = sum(1 for f in files if term in f.content)
        if count > 0:
            search_term = term
            break
    task = f'How many files contain "{search_term}"?'
    return _make_example(task, files, count)


_SUB_TYPES = [
    _count_by_extension,
    _count_by_name_pattern,
    _count_all_in_subtree,
    _count_files_containing,
]


def generate_file_counting_examples(
    n: int,
    rng: random.Random,
) -> list[Example]:
    """Generate n file-counting examples using the given RNG."""
    return [rng.choice(_SUB_TYPES)(rng) for _ in range(n)]
