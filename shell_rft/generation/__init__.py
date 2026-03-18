"""Task family generators."""

from __future__ import annotations

import random
from collections.abc import Callable

from shell_rft.generation.content_search import generate_content_search_examples
from shell_rft.generation.csv_filtering import generate_csv_filtering_examples
from shell_rft.generation.file_counting import generate_file_counting_examples
from shell_rft.generation.topk_by_size import generate_topk_by_size_examples
from shell_rft.schemas import Example

GENERATORS: dict[str, Callable[[int, random.Random], list[Example]]] = {
    "file_counting": generate_file_counting_examples,
    "content_search": generate_content_search_examples,
    "topk_by_size": generate_topk_by_size_examples,
    "csv_filtering": generate_csv_filtering_examples,
}
