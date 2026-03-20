"""Content-search task family generator."""

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
        files = build_workspace(rng)
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
        files = build_workspace(rng)
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
        files = build_workspace(rng)
        dirs = sorted({f.path.split("/")[0] for f in files})
        target_dir = rng.choice(dirs)
        dir_files = [f for f in files if f.path.startswith(f"{target_dir}/")]
        if len(dir_files) < 3:
            continue
        terms = list(_SEARCH_TERMS)
        rng.shuffle(terms)
        for term in terms:
            containing = [f for f in dir_files if term in f.content]
            not_containing = sorted(
                f.path for f in dir_files if term not in f.content
            )
            if containing and not_containing:
                task = (
                    f"Print the paths of files under {target_dir}/ that do NOT "
                    f"contain '{term}', one per line, sorted."
                )
                expected = "\n".join(not_containing) + "\n"
                return _make_example(task, files, expected)


def _count_matching_lines(rng: random.Random) -> Example:
    """Count total matching lines across files under a directory."""
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
            total = 0
            for f in dir_files:
                total += sum(1 for line in f.content.split("\n") if term in line)
            if total >= 2:
                task = (
                    f"How many lines in files under {target_dir}/ contain '{term}'?"
                )
                expected = f"{total}\n"
                return _make_example(task, files, expected)


def _find_files_containing_by_extension(rng: random.Random) -> Example:
    """Find files with a specific extension containing a search term."""
    while True:
        files = build_workspace(rng)
        dirs = sorted({f.path.split("/")[0] for f in files})
        target_dir = rng.choice(dirs)
        dir_files = [f for f in files if f.path.startswith(f"{target_dir}/")]
        exts = list({_file_ext(f.path) for f in dir_files if "." in f.path})
        rng.shuffle(exts)
        for ext in exts:
            ext_files = [f for f in dir_files if f.path.endswith(ext)]
            if len(ext_files) < 2:
                continue
            terms = list(_SEARCH_TERMS)
            rng.shuffle(terms)
            for term in terms:
                matches = sorted(
                    f.path for f in ext_files if term in f.content
                )
                non_matches = [f for f in ext_files if term not in f.content]
                if matches and non_matches:
                    task = (
                        f"Print the paths of {ext} files under {target_dir}/ "
                        f"that contain '{term}', one per line, sorted."
                    )
                    expected = "\n".join(matches) + "\n"
                    return _make_example(task, files, expected)


def _find_files_two_terms(rng: random.Random) -> Example:
    """Find files containing one term but not another."""
    while True:
        files = build_workspace(rng)
        dirs = sorted({f.path.split("/")[0] for f in files})
        target_dir = rng.choice(dirs)
        dir_files = [f for f in files if f.path.startswith(f"{target_dir}/")]
        if len(dir_files) < 4:
            continue
        terms = list(_SEARCH_TERMS)
        rng.shuffle(terms)
        for i, term_a in enumerate(terms):
            for term_b in terms[i + 1:]:
                matches = sorted(
                    f.path for f in dir_files
                    if term_a in f.content and term_b not in f.content
                )
                if len(matches) >= 1 and len(matches) < len(dir_files):
                    task = (
                        f"Print the paths of files under {target_dir}/ that "
                        f"contain '{term_a}' but do NOT contain '{term_b}', "
                        f"one per line, sorted."
                    )
                    expected = "\n".join(matches) + "\n"
                    return _make_example(task, files, expected)


def _count_files_containing(rng: random.Random) -> Example:
    """Count files containing a search term under a directory."""
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
            count = sum(1 for f in dir_files if term in f.content)
            if 0 < count < len(dir_files):
                task = (
                    f"How many files under {target_dir}/ contain '{term}'?"
                )
                expected = f"{count}\n"
                return _make_example(task, files, expected)


def _find_files_containing_excluding_subdir(rng: random.Random) -> Example:
    """Find files containing a term, excluding a subdirectory."""
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
            excluded = f"{top_dir}/{sub_dir}/"
            remaining = [
                f for f in files
                if f.path.startswith(f"{top_dir}/")
                and not f.path.startswith(excluded)
            ]
            if len(remaining) < 2:
                continue
            terms = list(_SEARCH_TERMS)
            rng.shuffle(terms)
            for term in terms:
                matches = sorted(f.path for f in remaining if term in f.content)
                if matches:
                    task = (
                        f"Print the paths of files under {top_dir}/ "
                        f"(excluding {top_dir}/{sub_dir}/) that contain "
                        f"'{term}', one per line, sorted."
                    )
                    expected = "\n".join(matches) + "\n"
                    return _make_example(task, files, expected)


_SUB_TYPES = [
    _find_files_containing,
    _grep_lines_from_file,
    _find_files_not_containing,
    _count_matching_lines,
    _find_files_containing_by_extension,
    _find_files_two_terms,
    _count_files_containing,
    _find_files_containing_excluding_subdir,
]


def generate_content_search_examples(
    n: int,
    rng: random.Random,
) -> list[Example]:
    """Generate n content-search examples using the given RNG."""
    return [rng.choice(_SUB_TYPES)(rng) for _ in range(n)]
