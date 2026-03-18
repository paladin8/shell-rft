"""Top-k by size task family generator."""

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


def _build_sized_workspace(rng: random.Random) -> list[FileSpec]:
    """Create files with distinct, known sizes."""
    n_dirs = rng.randint(2, 4)
    n_files = rng.randint(8, 15)
    dirs = rng.sample(_DIRS, n_dirs)

    sizes = rng.sample(range(100, 10001), n_files)

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

        content = "x" * (sizes[i] - 1) + "\n"
        files.append(FileSpec(path=path, content=content))

    return files


def _file_ext(path: str) -> str:
    """Return the file extension including the dot, e.g. '.py'."""
    dot = path.rfind(".")
    return path[dot:] if dot != -1 else ""


def _filesystem_summary_with_sizes(files: list[FileSpec]) -> str:
    return "\n".join(f"- {f.path} ({len(f.content)} bytes)" for f in files)


def _make_example(task: str, files: list[FileSpec], expected: str) -> Example:
    user_content = USER_PROMPT_TEMPLATE.format(
        task=task,
        filesystem_summary=_filesystem_summary_with_sizes(files),
    )
    return Example(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        workspace_spec=WorkspaceSpec(files=files),
        expected_stdout=expected,
        task_type="topk_by_size",
    )


def _largest_by_extension(rng: random.Random) -> Example:
    """K largest files of a given extension under a directory."""
    while True:
        files = _build_sized_workspace(rng)
        dirs = sorted({f.path.split("/")[0] for f in files})
        target_dir = rng.choice(dirs)
        dir_files = [f for f in files if f.path.startswith(f"{target_dir}/")]
        exts = list({_file_ext(f.path) for f in dir_files})
        rng.shuffle(exts)
        for ext in exts:
            matching = [f for f in dir_files if f.path.endswith(ext)]
            if len(matching) >= 2:
                matching.sort(key=lambda f: len(f.content), reverse=True)
                k = rng.randint(2, min(4, len(matching)))
                top_k = matching[:k]
                task = (
                    f"Print the {k} largest {ext} files under {target_dir}/, "
                    f"one per line, largest first."
                )
                expected = "\n".join(f.path for f in top_k) + "\n"
                return _make_example(task, files, expected)


def _largest_in_dir(rng: random.Random) -> Example:
    """K largest files (any extension) under a directory."""
    while True:
        files = _build_sized_workspace(rng)
        dirs = sorted({f.path.split("/")[0] for f in files})
        target_dir = rng.choice(dirs)
        matching = [f for f in files if f.path.startswith(f"{target_dir}/")]
        if len(matching) >= 2:
            matching.sort(key=lambda f: len(f.content), reverse=True)
            k = rng.randint(2, min(4, len(matching)))
            top_k = matching[:k]
            task = (
                f"Print the {k} largest files under {target_dir}/, "
                f"one per line, largest first."
            )
            expected = "\n".join(f.path for f in top_k) + "\n"
            return _make_example(task, files, expected)


def _smallest_in_dir(rng: random.Random) -> Example:
    """K smallest files under a directory."""
    while True:
        files = _build_sized_workspace(rng)
        dirs = sorted({f.path.split("/")[0] for f in files})
        target_dir = rng.choice(dirs)
        matching = [f for f in files if f.path.startswith(f"{target_dir}/")]
        if len(matching) >= 2:
            matching.sort(key=lambda f: len(f.content))
            k = rng.randint(2, min(4, len(matching)))
            bottom_k = matching[:k]
            task = (
                f"Print the {k} smallest files under {target_dir}/, "
                f"one per line, smallest first."
            )
            expected = "\n".join(f.path for f in bottom_k) + "\n"
            return _make_example(task, files, expected)


_SUB_TYPES = [
    _largest_by_extension,
    _largest_in_dir,
    _smallest_in_dir,
]


def generate_topk_by_size_examples(
    n: int,
    rng: random.Random,
) -> list[Example]:
    """Generate n top-k-by-size examples using the given RNG."""
    return [rng.choice(_SUB_TYPES)(rng) for _ in range(n)]
