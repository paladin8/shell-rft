"""Shared workspace builder for task generators."""

from __future__ import annotations

import random

from shell_rft.schemas import FileSpec

_DIRS = ["src", "logs", "data", "app", "lib", "tests", "docs", "config"]
_SUBDIRS = ["utils", "core", "api", "auth", "models", "services", "handlers"]
_DEEP_SUBDIRS = ["v1", "v2", "legacy", "internal", "helpers"]
_EXTENSIONS = [".py", ".txt", ".log", ".csv", ".json", ".yaml", ".md", ".sh"]
_STEMS = [
    "main", "utils", "config", "app", "debug", "error",
    "output", "report", "index", "helper", "server", "client",
    "handler", "router", "schema", "validator", "parser", "worker",
]
_CONTENT_LINES = [
    "# placeholder",
    "import os",
    "import sys",
    "from pathlib import Path",
    "ERROR: connection refused",
    "ERROR: timeout exceeded",
    "ERROR: disk full",
    "DEBUG: starting service",
    "DEBUG: request received",
    "DEBUG: cache hit",
    "name,value",
    "hello world",
    "TODO: fix this",
    "TODO: refactor later",
    "print('hello')",
    "status=ok",
    "timeout exceeded",
    "WARNING: disk space low",
    "WARNING: deprecated API",
    "return None",
    "raise ValueError",
    "class Config:",
    "def setup():",
    "import logging",
]
_SEARCH_TERMS = ["ERROR", "TODO", "import", "timeout", "hello", "DEBUG", "WARNING"]


def build_workspace(
    rng: random.Random,
    *,
    n_files: tuple[int, int] = (20, 40),
    n_dirs: tuple[int, int] = (3, 6),
    depth_prob: float = 0.5,
    deep_depth_prob: float = 0.15,
    test_prefix_prob: float = 0.2,
) -> list[FileSpec]:
    """Create a random set of files across a directory tree.

    Args:
        n_files: (min, max) number of files to generate.
        n_dirs: (min, max) number of top-level directories.
        depth_prob: probability of nesting one level deeper.
        deep_depth_prob: probability of nesting two levels deeper.
        test_prefix_prob: probability of a test_ prefix on filenames.
    """
    num_dirs = rng.randint(*n_dirs)
    num_files = rng.randint(*n_files)
    dirs = rng.sample(_DIRS, min(num_dirs, len(_DIRS)))

    files: list[FileSpec] = []
    seen: set[str] = set()

    for i in range(num_files):
        d = rng.choice(dirs)
        r = rng.random()
        if r < deep_depth_prob:
            d = f"{d}/{rng.choice(_SUBDIRS)}/{rng.choice(_DEEP_SUBDIRS)}"
        elif r < depth_prob:
            d = f"{d}/{rng.choice(_SUBDIRS)}"

        stem = rng.choice(_STEMS)
        ext = rng.choice(_EXTENSIONS)
        prefix = "test_" if rng.random() < test_prefix_prob else ""
        path = f"{d}/{prefix}{stem}{ext}"

        if path in seen:
            path = f"{d}/{prefix}{stem}_{i}{ext}"
        if path in seen:
            continue
        seen.add(path)

        n_lines = rng.randint(2, 8)
        content = "\n".join(rng.choices(_CONTENT_LINES, k=n_lines)) + "\n"
        files.append(FileSpec(path=path, content=content))

    return files
