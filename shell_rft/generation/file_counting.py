"""File-counting task family generator."""

from __future__ import annotations

import random

from shell_rft.generation.workspace import (
    _SEARCH_TERMS,
    build_workspace,
)
from shell_rft.prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE
from shell_rft.schemas import Example, FileSpec, WorkspaceSpec


def _file_ext(path: str) -> str:
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
    files = build_workspace(rng)
    exts_present = sorted({_file_ext(f.path) for f in files})
    ext = rng.choice(exts_present)
    dirs = sorted({f.path.split("/")[0] for f in files})
    target_dir = rng.choice(dirs)
    count = sum(
        1 for f in files
        if f.path.startswith(f"{target_dir}/") and f.path.endswith(ext)
    )
    if count == 0:
        count = sum(1 for f in files if f.path.endswith(ext))
        task = f"Count the {ext} files."
    else:
        task = f"Count the {ext} files under {target_dir}/."
    return _make_example(task, files, count)


def _count_by_name_pattern(rng: random.Random) -> Example:
    while True:
        files = build_workspace(rng)
        count = sum(
            1 for f in files
            if f.path.split("/")[-1].startswith("test_")
        )
        if count > 0:
            break
    task = 'Count files whose name starts with "test_".'
    return _make_example(task, files, count)


def _count_all_in_subtree(rng: random.Random) -> Example:
    files = build_workspace(rng)
    dirs = sorted({f.path.split("/")[0] for f in files})
    target_dir = rng.choice(dirs)
    count = sum(1 for f in files if f.path.startswith(f"{target_dir}/"))
    task = f"How many files are in {target_dir}/?"
    return _make_example(task, files, count)


def _count_files_containing(rng: random.Random) -> Example:
    files = build_workspace(rng)
    terms = list(_SEARCH_TERMS)
    rng.shuffle(terms)
    for term in terms:
        count = sum(1 for f in files if term in f.content)
        if count > 0:
            task = f'How many files contain "{term}"?'
            return _make_example(task, files, count)
    return _count_all_in_subtree(rng)


def _count_by_extension_excluding_subdir(rng: random.Random) -> Example:
    """Count files by extension, excluding a subdirectory."""
    while True:
        files = build_workspace(rng)
        subdir_pairs: list[tuple[str, str]] = []
        for f in files:
            parts = f.path.split("/")
            if len(parts) >= 3:
                pair = (parts[0], parts[1])
                if pair not in subdir_pairs:
                    subdir_pairs.append(pair)
        if not subdir_pairs:
            continue
        rng.shuffle(subdir_pairs)
        for top_dir, sub_dir in subdir_pairs:
            excluded_prefix = f"{top_dir}/{sub_dir}/"
            dir_files = [
                f for f in files
                if f.path.startswith(f"{top_dir}/")
                and not f.path.startswith(excluded_prefix)
            ]
            exts = sorted({_file_ext(f.path) for f in dir_files})
            rng.shuffle(exts)
            for ext in exts:
                count = sum(1 for f in dir_files if f.path.endswith(ext))
                if count >= 1:
                    task = (
                        f"Count the {ext} files under {top_dir}/, "
                        f"excluding {top_dir}/{sub_dir}/."
                    )
                    return _make_example(task, files, count)


def _count_by_extension_and_content(rng: random.Random) -> Example:
    """Count files matching both an extension and a content pattern."""
    while True:
        files = build_workspace(rng)
        dirs = sorted({f.path.split("/")[0] for f in files})
        target_dir = rng.choice(dirs)
        dir_files = [f for f in files if f.path.startswith(f"{target_dir}/")]
        exts = sorted({_file_ext(f.path) for f in dir_files})
        rng.shuffle(exts)
        for ext in exts:
            ext_files = [f for f in dir_files if f.path.endswith(ext)]
            if len(ext_files) < 2:
                continue
            terms = list(_SEARCH_TERMS)
            rng.shuffle(terms)
            for term in terms:
                count = sum(1 for f in ext_files if term in f.content)
                if 0 < count < len(ext_files):
                    task = (
                        f"Count the {ext} files under {target_dir}/ "
                        f'that contain "{term}".'
                    )
                    return _make_example(task, files, count)


def _count_unique_extensions(rng: random.Random) -> Example:
    """Count how many distinct file extensions exist under a directory."""
    files = build_workspace(rng)
    dirs = sorted({f.path.split("/")[0] for f in files})
    target_dir = rng.choice(dirs)
    dir_files = [f for f in files if f.path.startswith(f"{target_dir}/")]
    exts = {_file_ext(f.path) for f in dir_files if _file_ext(f.path)}
    count = len(exts)
    if count < 2:
        exts = {_file_ext(f.path) for f in files if _file_ext(f.path)}
        count = len(exts)
        task = "How many distinct file extensions exist in the workspace?"
    else:
        task = f"How many distinct file extensions exist under {target_dir}/?"
    return _make_example(task, files, count)


def _count_empty_files(rng: random.Random) -> Example:
    """Count files with only whitespace/empty content under a directory."""
    files = build_workspace(rng)
    # Replace some files with empty content
    for f in files:
        if rng.random() < 0.2:
            f.content = "\n"
    dirs = sorted({f.path.split("/")[0] for f in files})
    target_dir = rng.choice(dirs)
    dir_files = [f for f in files if f.path.startswith(f"{target_dir}/")]
    count = sum(1 for f in dir_files if f.content.strip() == "")
    if count == 0:
        return _count_all_in_subtree(rng)
    task = f"How many empty files are under {target_dir}/?"
    return _make_example(task, files, count)


def _count_files_not_containing(rng: random.Random) -> Example:
    """Count files that do NOT contain a search term."""
    while True:
        files = build_workspace(rng)
        dirs = sorted({f.path.split("/")[0] for f in files})
        target_dir = rng.choice(dirs)
        dir_files = [f for f in files if f.path.startswith(f"{target_dir}/")]
        if len(dir_files) < 3:
            continue
        terms = list(_SEARCH_TERMS)
        rng.shuffle(terms)
        for term in terms:
            has = sum(1 for f in dir_files if term in f.content)
            not_has = len(dir_files) - has
            if has >= 1 and not_has >= 1:
                task = (
                    f'How many files under {target_dir}/ do NOT contain "{term}"?'
                )
                return _make_example(task, files, not_has)


_SUB_TYPES = [
    _count_by_extension,
    _count_by_name_pattern,
    _count_all_in_subtree,
    _count_files_containing,
    _count_by_extension_excluding_subdir,
    _count_by_extension_and_content,
    _count_unique_extensions,
    _count_empty_files,
    _count_files_not_containing,
]


def generate_file_counting_examples(
    n: int,
    rng: random.Random,
) -> list[Example]:
    """Generate n file-counting examples using the given RNG."""
    return [rng.choice(_SUB_TYPES)(rng) for _ in range(n)]
