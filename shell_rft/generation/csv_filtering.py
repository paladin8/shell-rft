"""CSV filtering task family generator."""

from __future__ import annotations

import random

from shell_rft.prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE
from shell_rft.schemas import Example, FileSpec, WorkspaceSpec

_CSV_SCHEMAS = [
    {
        "path": "data/users.csv",
        "columns": ["name", "role", "usage_pct"],
        "names": [
            "alice", "bob", "charlie", "diana",
            "eve", "frank", "grace", "heidi",
        ],
        "categories": ["admin", "user", "viewer"],
        "numeric_range": (10, 99),
    },
    {
        "path": "data/servers.csv",
        "columns": ["host", "status", "response_ms"],
        "names": [
            "web01", "web02", "db01", "db02",
            "cache01", "api01", "api02", "worker01",
        ],
        "categories": ["up", "down", "degraded"],
        "numeric_range": (50, 2000),
    },
    {
        "path": "data/jobs.csv",
        "columns": ["job_id", "priority", "duration_s"],
        "names": [
            "job_001", "job_002", "job_003", "job_004",
            "job_005", "job_006", "job_007", "job_008",
        ],
        "categories": ["high", "medium", "low"],
        "numeric_range": (1, 500),
    },
]


def _generate_csv_data(
    schema: dict, rng: random.Random,
) -> tuple[str, list[list[str]]]:
    """Generate CSV content. Returns (content_string, data_rows_as_lists)."""
    n_rows = rng.randint(5, 12)

    names = list(schema["names"])
    rng.shuffle(names)
    names = names[:n_rows]
    # Pad if not enough unique names.
    while len(names) < n_rows:
        names.append(f"{rng.choice(schema['names'])}_{len(names)}")

    rows: list[list[str]] = []
    lines: list[str] = []
    for name in names:
        category = rng.choice(schema["categories"])
        numeric = rng.randint(*schema["numeric_range"])
        row = [name, category, str(numeric)]
        rows.append(row)
        lines.append(",".join(row))

    content = "\n".join(lines) + "\n"
    return content, rows


def _pick_threshold(values: list[int], rng: random.Random) -> int:
    """Pick a threshold that guarantees at least one value exceeds it."""
    sorted_vals = sorted(set(values))
    split_idx = rng.randint(0, len(sorted_vals) - 1)
    return sorted_vals[split_idx] - 1


def _filesystem_summary(files: list[FileSpec], schema: dict) -> str:
    columns = ",".join(schema["columns"])
    summaries = []
    for f in files:
        if f.path.endswith(".csv"):
            summaries.append(f"- {f.path} (columns: {columns})")
        else:
            summaries.append(f"- {f.path}")
    return "\n".join(summaries)


def _make_example(
    task: str, files: list[FileSpec], expected: str, schema: dict,
) -> Example:
    user_content = USER_PROMPT_TEMPLATE.format(
        task=task,
        filesystem_summary=_filesystem_summary(files, schema),
    )
    return Example(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        workspace_spec=WorkspaceSpec(files=files),
        expected_stdout=expected,
        task_type="csv_filtering",
    )


def _numeric_filter(rng: random.Random) -> Example:
    """Print column values where a numeric column exceeds a threshold."""
    schema = rng.choice(_CSV_SCHEMAS)
    content, rows = _generate_csv_data(schema, rng)
    columns = schema["columns"]
    numeric_col = columns[2]
    output_col = columns[0]

    values = [int(row[2]) for row in rows]
    threshold = _pick_threshold(values, rng)
    matching = [row[0] for row in rows if int(row[2]) > threshold]

    path = schema["path"]
    task = (
        f"Print the {output_col} values from {path} where "
        f"{numeric_col} is greater than {threshold}."
    )
    expected = "\n".join(matching) + "\n"
    files = [FileSpec(path=path, content=content)]
    return _make_example(task, files, expected, schema)


def _string_filter(rng: random.Random) -> Example:
    """Print column values where a string column matches a value."""
    schema = rng.choice(_CSV_SCHEMAS)
    content, rows = _generate_csv_data(schema, rng)
    columns = schema["columns"]
    string_col = columns[1]
    output_col = columns[0]

    # Pick a category value that appears in the data.
    present_categories = list({row[1] for row in rows})
    filter_value = rng.choice(present_categories)
    matching = [row[0] for row in rows if row[1] == filter_value]

    path = schema["path"]
    task = (
        f"Print the {output_col} values from {path} where "
        f"{string_col} is '{filter_value}'."
    )
    expected = "\n".join(matching) + "\n"
    files = [FileSpec(path=path, content=content)]
    return _make_example(task, files, expected, schema)


def _count_by_numeric(rng: random.Random) -> Example:
    """Count rows where a numeric column exceeds a threshold."""
    schema = rng.choice(_CSV_SCHEMAS)
    content, rows = _generate_csv_data(schema, rng)
    columns = schema["columns"]
    numeric_col = columns[2]

    values = [int(row[2]) for row in rows]
    threshold = _pick_threshold(values, rng)
    count = sum(1 for row in rows if int(row[2]) > threshold)

    path = schema["path"]
    task = (
        f"How many rows in {path} have {numeric_col} greater than {threshold}?"
    )
    expected = f"{count}\n"
    files = [FileSpec(path=path, content=content)]
    return _make_example(task, files, expected, schema)


_SUB_TYPES = [
    _numeric_filter,
    _string_filter,
    _count_by_numeric,
]


def generate_csv_filtering_examples(
    n: int,
    rng: random.Random,
) -> list[Example]:
    """Generate n csv-filtering examples using the given RNG."""
    return [rng.choice(_SUB_TYPES)(rng) for _ in range(n)]
