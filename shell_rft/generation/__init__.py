"""Task family generators."""

from __future__ import annotations

import random
from collections.abc import Callable

from shell_rft.generation.file_counting import generate_file_counting_examples
from shell_rft.schemas import Example

GENERATORS: dict[str, Callable[[int, random.Random], list[Example]]] = {
    "file_counting": generate_file_counting_examples,
}
