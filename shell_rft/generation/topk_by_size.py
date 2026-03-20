"""Top-k by size task family generator."""

from __future__ import annotations

import random

from shell_rft.generation.workspace import _DIRS, _SUBDIRS, _DEEP_SUBDIRS
from shell_rft.prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE
from shell_rft.schemas import Example, FileSpec, WorkspaceSpec

_EXTENSIONS = [".py", ".txt", ".log", ".csv", ".json", ".yaml", ".md"]
_STEMS = [
    "main", "utils", "config", "app", "debug", "error",
    "output", "report", "index", "helper", "server", "client",
    "handler", "router", "schema", "worker",
]


def _build_sized_workspace(rng: random.Random) -> list[FileSpec]:
    """Create files with distinct, known sizes across a deeper tree."""
    n_dirs = rng.randint(3, 6)
    n_files = rng.randint(15, 30)
    dirs = rng.sample(_DIRS, min(n_dirs, len(_DIRS)))

    sizes = rng.sample(range(100, 10001), n_files)

    files: list[FileSpec] = []
    seen: set[str] = set()

    for i in range(n_files):
        d = rng.choice(dirs)
        r = rng.random()
        if r < 0.1:
            d = f"{d}/{rng.choice(_SUBDIRS)}/{rng.choice(_DEEP_SUBDIRS)}"
        elif r < 0.45:
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


def _ordinal(n: int) -> str:
    """Return ordinal string for n, e.g. 1 -> '1st', 3 -> '3rd'."""
    if 11 <= n % 100 <= 13:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def _largest_excluding_subdir(rng: random.Random) -> Example:
    """K largest files under a directory, excluding a subdirectory."""
    while True:
        files = _build_sized_workspace(rng)
        # Find (top_dir, sub_dir) pairs present in the workspace.
        subdir_pairs: list[tuple[str, str]] = []
        seen_pairs: set[tuple[str, str]] = set()
        for f in files:
            parts = f.path.split("/")
            if len(parts) >= 3:
                pair = (parts[0], parts[1])
                if pair not in seen_pairs:
                    seen_pairs.add(pair)
                    subdir_pairs.append(pair)

        rng.shuffle(subdir_pairs)
        for top_dir, sub_dir in subdir_pairs:
            excluded_prefix = f"{top_dir}/{sub_dir}/"
            remaining = [
                f for f in files
                if f.path.startswith(f"{top_dir}/")
                and not f.path.startswith(excluded_prefix)
            ]
            excluded = [
                f for f in files if f.path.startswith(excluded_prefix)
            ]
            if len(remaining) >= 2 and len(excluded) >= 1:
                remaining.sort(key=lambda f: len(f.content), reverse=True)
                k = rng.randint(2, min(4, len(remaining)))
                top_k = remaining[:k]
                task = (
                    f"Print the {k} largest files under {top_dir}/, "
                    f"excluding {top_dir}/{sub_dir}/, "
                    f"one per line, largest first."
                )
                expected = "\n".join(f.path for f in top_k) + "\n"
                return _make_example(task, files, expected)


def _range_by_size(rng: random.Random) -> Example:
    """Files ranked Nth through Mth by size under a directory."""
    while True:
        files = _build_sized_workspace(rng)
        dirs = sorted({f.path.split("/")[0] for f in files})
        target_dir = rng.choice(dirs)
        matching = [f for f in files if f.path.startswith(f"{target_dir}/")]
        if len(matching) >= 5:
            matching.sort(key=lambda f: len(f.content), reverse=True)
            start = rng.randint(2, len(matching) - 2)
            end = rng.randint(start + 1, min(start + 3, len(matching)))
            range_files = matching[start - 1 : end]
            task = (
                f"Print the {_ordinal(start)} through {_ordinal(end)} largest "
                f"files under {target_dir}/, one per line, largest first."
            )
            expected = "\n".join(f.path for f in range_files) + "\n"
            return _make_example(task, files, expected)


_SUB_TYPES = [
    _largest_by_extension,
    _largest_in_dir,
    _smallest_in_dir,
    _largest_excluding_subdir,
    _range_by_size,
]


def generate_topk_by_size_examples(
    n: int,
    rng: random.Random,
) -> list[Example]:
    """Generate n top-k-by-size examples using the given RNG."""
    return [rng.choice(_SUB_TYPES)(rng) for _ in range(n)]
