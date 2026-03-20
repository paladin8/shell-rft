"""CSV filtering task family generator."""

from __future__ import annotations

import random

from shell_rft.prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE
from shell_rft.schemas import Example, FileSpec, WorkspaceSpec

_CSV_SCHEMAS = [
    {
        "path": "data/users.csv",
        "columns": ["name", "role", "department", "usage_pct", "login_count"],
        "names": [
            "alice", "bob", "charlie", "diana", "eve", "frank",
            "grace", "heidi", "ivan", "judy", "karl", "laura",
            "mallory", "nancy", "oscar", "peggy",
        ],
        "categories": {
            "role": ["admin", "user", "viewer", "editor"],
            "department": ["engineering", "sales", "support", "marketing"],
        },
        "numeric_columns": {
            "usage_pct": (5, 99),
            "login_count": (0, 500),
        },
    },
    {
        "path": "data/servers.csv",
        "columns": ["host", "status", "region", "response_ms", "cpu_pct"],
        "names": [
            "web01", "web02", "web03", "db01", "db02",
            "cache01", "cache02", "api01", "api02", "api03",
            "worker01", "worker02", "lb01", "lb02", "monitor01",
        ],
        "categories": {
            "status": ["up", "down", "degraded", "maintenance"],
            "region": ["us-east", "us-west", "eu-west", "ap-south"],
        },
        "numeric_columns": {
            "response_ms": (10, 3000),
            "cpu_pct": (1, 99),
        },
    },
    {
        "path": "data/jobs.csv",
        "columns": ["job_id", "priority", "status", "duration_s", "retries"],
        "names": [
            "job_001", "job_002", "job_003", "job_004", "job_005",
            "job_006", "job_007", "job_008", "job_009", "job_010",
            "job_011", "job_012", "job_013", "job_014", "job_015",
        ],
        "categories": {
            "priority": ["critical", "high", "medium", "low"],
            "status": ["running", "completed", "failed", "queued"],
        },
        "numeric_columns": {
            "duration_s": (1, 1000),
            "retries": (0, 10),
        },
    },
]


def _generate_csv_data(
    schema: dict, rng: random.Random, n_rows: int | None = None,
) -> tuple[str, list[list[str]]]:
    """Generate CSV content with a header row.

    Returns (content_string, data_rows_as_lists) where data_rows
    excludes the header.
    """
    if n_rows is None:
        n_rows = rng.randint(12, 25)

    columns = schema["columns"]
    names = list(schema["names"])
    rng.shuffle(names)
    names = names[:n_rows]
    while len(names) < n_rows:
        names.append(f"{rng.choice(schema['names'])}_{len(names)}")

    rows: list[list[str]] = []
    header = ",".join(columns)
    lines: list[str] = [header]
    for name in names:
        row = [name]
        for col in columns[1:]:
            if col in schema.get("categories", {}):
                row.append(rng.choice(schema["categories"][col]))
            elif col in schema.get("numeric_columns", {}):
                lo, hi = schema["numeric_columns"][col]
                row.append(str(rng.randint(lo, hi)))
            else:
                row.append("unknown")
        rows.append(row)
        lines.append(",".join(row))

    content = "\n".join(lines) + "\n"
    return content, rows


def _col_index(schema: dict, col_name: str) -> int:
    return schema["columns"].index(col_name)


def _pick_threshold(values: list[int], rng: random.Random) -> int:
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


# --- Sub-types ---


def _numeric_filter(rng: random.Random) -> Example:
    """Print column values where a numeric column exceeds a threshold."""
    schema = rng.choice(_CSV_SCHEMAS)
    content, rows = _generate_csv_data(schema, rng)
    num_cols = list(schema["numeric_columns"].keys())
    num_col = rng.choice(num_cols)
    col_idx = _col_index(schema, num_col)
    output_col = schema["columns"][0]

    values = [int(row[col_idx]) for row in rows]
    threshold = _pick_threshold(values, rng)
    matching = [row[0] for row in rows if int(row[col_idx]) > threshold]

    path = schema["path"]
    task = (
        f"Print the {output_col} values from {path} where "
        f"{num_col} is greater than {threshold}."
    )
    expected = "\n".join(matching) + "\n"
    files = [FileSpec(path=path, content=content)]
    return _make_example(task, files, expected, schema)


def _string_filter(rng: random.Random) -> Example:
    """Print column values where a string column matches a value."""
    schema = rng.choice(_CSV_SCHEMAS)
    content, rows = _generate_csv_data(schema, rng)
    cat_cols = list(schema["categories"].keys())
    cat_col = rng.choice(cat_cols)
    col_idx = _col_index(schema, cat_col)
    output_col = schema["columns"][0]

    present = list({row[col_idx] for row in rows})
    filter_value = rng.choice(present)
    matching = [row[0] for row in rows if row[col_idx] == filter_value]

    path = schema["path"]
    task = (
        f"Print the {output_col} values from {path} where "
        f"{cat_col} is '{filter_value}'."
    )
    expected = "\n".join(matching) + "\n"
    files = [FileSpec(path=path, content=content)]
    return _make_example(task, files, expected, schema)


def _count_by_numeric(rng: random.Random) -> Example:
    """Count rows where a numeric column exceeds a threshold."""
    schema = rng.choice(_CSV_SCHEMAS)
    content, rows = _generate_csv_data(schema, rng)
    num_cols = list(schema["numeric_columns"].keys())
    num_col = rng.choice(num_cols)
    col_idx = _col_index(schema, num_col)

    values = [int(row[col_idx]) for row in rows]
    threshold = _pick_threshold(values, rng)
    count = sum(1 for row in rows if int(row[col_idx]) > threshold)

    path = schema["path"]
    task = f"How many rows in {path} have {num_col} greater than {threshold}?"
    expected = f"{count}\n"
    files = [FileSpec(path=path, content=content)]
    return _make_example(task, files, expected, schema)


def _multi_condition_filter(rng: random.Random) -> Example:
    """Filter by both string and numeric conditions."""
    schema = rng.choice(_CSV_SCHEMAS)
    content, rows = _generate_csv_data(schema, rng)
    cat_cols = list(schema["categories"].keys())
    num_cols = list(schema["numeric_columns"].keys())
    cat_col = rng.choice(cat_cols)
    num_col = rng.choice(num_cols)
    cat_idx = _col_index(schema, cat_col)
    num_idx = _col_index(schema, num_col)
    output_col = schema["columns"][0]

    present = list({row[cat_idx] for row in rows})
    filter_value = rng.choice(present)
    category_rows = [row for row in rows if row[cat_idx] == filter_value]
    if not category_rows:
        return _numeric_filter(rng)

    values = [int(row[num_idx]) for row in category_rows]
    threshold = _pick_threshold(values, rng)
    matching = [
        row[0] for row in rows
        if row[cat_idx] == filter_value and int(row[num_idx]) > threshold
    ]
    if not matching:
        return _numeric_filter(rng)

    path = schema["path"]
    task = (
        f"Print the {output_col} values from {path} where "
        f"{cat_col} is '{filter_value}' and {num_col} is greater than {threshold}."
    )
    expected = "\n".join(matching) + "\n"
    files = [FileSpec(path=path, content=content)]
    return _make_example(task, files, expected, schema)


def _sorted_filter(rng: random.Random) -> Example:
    """Filter and sort the output alphabetically."""
    schema = rng.choice(_CSV_SCHEMAS)
    content, rows = _generate_csv_data(schema, rng)
    cat_cols = list(schema["categories"].keys())
    cat_col = rng.choice(cat_cols)
    col_idx = _col_index(schema, cat_col)
    output_col = schema["columns"][0]

    present = list({row[col_idx] for row in rows})
    filter_value = rng.choice(present)
    matching = sorted(row[0] for row in rows if row[col_idx] == filter_value)

    path = schema["path"]
    task = (
        f"Print the {output_col} values from {path} where "
        f"{cat_col} is '{filter_value}', sorted alphabetically."
    )
    expected = "\n".join(matching) + "\n"
    files = [FileSpec(path=path, content=content)]
    return _make_example(task, files, expected, schema)


def _count_by_category(rng: random.Random) -> Example:
    """Count rows matching a string filter."""
    schema = rng.choice(_CSV_SCHEMAS)
    content, rows = _generate_csv_data(schema, rng)
    cat_cols = list(schema["categories"].keys())
    cat_col = rng.choice(cat_cols)
    col_idx = _col_index(schema, cat_col)

    present = list({row[col_idx] for row in rows})
    filter_value = rng.choice(present)
    count = sum(1 for row in rows if row[col_idx] == filter_value)

    path = schema["path"]
    task = f"How many rows in {path} have {cat_col} equal to '{filter_value}'?"
    expected = f"{count}\n"
    files = [FileSpec(path=path, content=content)]
    return _make_example(task, files, expected, schema)


def _max_min_aggregate(rng: random.Random) -> Example:
    """Find the row with max or min value in a numeric column."""
    schema = rng.choice(_CSV_SCHEMAS)
    content, rows = _generate_csv_data(schema, rng)
    num_cols = list(schema["numeric_columns"].keys())
    num_col = rng.choice(num_cols)
    col_idx = _col_index(schema, num_col)
    output_col = schema["columns"][0]
    path = schema["path"]

    if rng.random() < 0.5:
        best_row = max(rows, key=lambda r: int(r[col_idx]))
        task = f"Print the {output_col} from {path} with the highest {num_col}."
    else:
        best_row = min(rows, key=lambda r: int(r[col_idx]))
        task = f"Print the {output_col} from {path} with the lowest {num_col}."

    expected = best_row[0] + "\n"
    files = [FileSpec(path=path, content=content)]
    return _make_example(task, files, expected, schema)


def _print_non_id_column(rng: random.Random) -> Example:
    """Print a non-identifier column filtered by another column."""
    schema = rng.choice(_CSV_SCHEMAS)
    content, rows = _generate_csv_data(schema, rng)
    cat_cols = list(schema["categories"].keys())
    cat_col = rng.choice(cat_cols)
    cat_idx = _col_index(schema, cat_col)

    # Pick a different column to output (not the identifier, not the filter)
    other_cols = [c for c in schema["columns"] if c != schema["columns"][0] and c != cat_col]
    if not other_cols:
        return _string_filter(rng)
    output_col = rng.choice(other_cols)
    out_idx = _col_index(schema, output_col)

    present = list({row[cat_idx] for row in rows})
    filter_value = rng.choice(present)
    matching = [row[out_idx] for row in rows if row[cat_idx] == filter_value]

    path = schema["path"]
    task = (
        f"Print the {output_col} values from {path} where "
        f"{cat_col} is '{filter_value}'."
    )
    expected = "\n".join(matching) + "\n"
    files = [FileSpec(path=path, content=content)]
    return _make_example(task, files, expected, schema)


def _count_multi_condition(rng: random.Random) -> Example:
    """Count rows matching both a string and numeric condition."""
    schema = rng.choice(_CSV_SCHEMAS)
    content, rows = _generate_csv_data(schema, rng)
    cat_cols = list(schema["categories"].keys())
    num_cols = list(schema["numeric_columns"].keys())
    cat_col = rng.choice(cat_cols)
    num_col = rng.choice(num_cols)
    cat_idx = _col_index(schema, cat_col)
    num_idx = _col_index(schema, num_col)

    present = list({row[cat_idx] for row in rows})
    filter_value = rng.choice(present)
    category_rows = [row for row in rows if row[cat_idx] == filter_value]
    if not category_rows:
        return _count_by_numeric(rng)

    values = [int(row[num_idx]) for row in category_rows]
    threshold = _pick_threshold(values, rng)
    count = sum(
        1 for row in rows
        if row[cat_idx] == filter_value and int(row[num_idx]) > threshold
    )
    if count == 0:
        return _count_by_numeric(rng)

    path = schema["path"]
    task = (
        f"How many rows in {path} have {cat_col} equal to '{filter_value}' "
        f"and {num_col} greater than {threshold}?"
    )
    expected = f"{count}\n"
    files = [FileSpec(path=path, content=content)]
    return _make_example(task, files, expected, schema)


def _top_n_by_column(rng: random.Random) -> Example:
    """Print the top N rows by a numeric column (requires sort pipeline)."""
    schema = rng.choice(_CSV_SCHEMAS)
    content, rows = _generate_csv_data(schema, rng)
    num_cols = list(schema["numeric_columns"].keys())
    num_col = rng.choice(num_cols)
    col_idx = _col_index(schema, num_col)
    output_col = schema["columns"][0]
    path = schema["path"]

    n = rng.randint(2, 4)
    sorted_rows = sorted(rows, key=lambda r: int(r[col_idx]), reverse=True)
    top_n = sorted_rows[:n]

    task = (
        f"Print the {output_col} of the {n} rows in {path} with the "
        f"highest {num_col}, one per line, highest first."
    )
    expected = "\n".join(row[0] for row in top_n) + "\n"
    files = [FileSpec(path=path, content=content)]
    return _make_example(task, files, expected, schema)


def _count_distinct(rng: random.Random) -> Example:
    """Count distinct values in a categorical column."""
    schema = rng.choice(_CSV_SCHEMAS)
    content, rows = _generate_csv_data(schema, rng)
    cat_cols = list(schema["categories"].keys())
    cat_col = rng.choice(cat_cols)
    col_idx = _col_index(schema, cat_col)
    path = schema["path"]

    distinct = len({row[col_idx] for row in rows})
    task = f"How many distinct {cat_col} values appear in {path}?"
    expected = f"{distinct}\n"
    files = [FileSpec(path=path, content=content)]
    return _make_example(task, files, expected, schema)


def _sum_numeric(rng: random.Random) -> Example:
    """Sum a numeric column, optionally filtered."""
    schema = rng.choice(_CSV_SCHEMAS)
    content, rows = _generate_csv_data(schema, rng)
    num_cols = list(schema["numeric_columns"].keys())
    num_col = rng.choice(num_cols)
    col_idx = _col_index(schema, num_col)
    path = schema["path"]

    if rng.random() < 0.5:
        # Sum all
        total = sum(int(row[col_idx]) for row in rows)
        task = f"What is the sum of {num_col} across all rows in {path}?"
    else:
        # Sum filtered by category
        cat_cols = list(schema["categories"].keys())
        cat_col = rng.choice(cat_cols)
        cat_idx = _col_index(schema, cat_col)
        present = list({row[cat_idx] for row in rows})
        filter_value = rng.choice(present)
        total = sum(
            int(row[col_idx]) for row in rows if row[cat_idx] == filter_value
        )
        task = (
            f"What is the sum of {num_col} for rows in {path} where "
            f"{cat_col} is '{filter_value}'?"
        )

    expected = f"{total}\n"
    files = [FileSpec(path=path, content=content)]
    return _make_example(task, files, expected, schema)


def _filter_two_categories(rng: random.Random) -> Example:
    """Filter by two different categorical columns simultaneously."""
    schema = rng.choice(_CSV_SCHEMAS)
    if len(schema["categories"]) < 2:
        return _multi_condition_filter(rng)
    content, rows = _generate_csv_data(schema, rng)
    cat_cols = list(schema["categories"].keys())
    rng.shuffle(cat_cols)
    cat_col_a, cat_col_b = cat_cols[0], cat_cols[1]
    idx_a = _col_index(schema, cat_col_a)
    idx_b = _col_index(schema, cat_col_b)
    output_col = schema["columns"][0]

    present_a = list({row[idx_a] for row in rows})
    present_b = list({row[idx_b] for row in rows})
    val_a = rng.choice(present_a)
    val_b = rng.choice(present_b)
    matching = [row[0] for row in rows if row[idx_a] == val_a and row[idx_b] == val_b]
    if not matching:
        return _multi_condition_filter(rng)

    path = schema["path"]
    task = (
        f"Print the {output_col} values from {path} where "
        f"{cat_col_a} is '{val_a}' and {cat_col_b} is '{val_b}'."
    )
    expected = "\n".join(matching) + "\n"
    files = [FileSpec(path=path, content=content)]
    return _make_example(task, files, expected, schema)


def _numeric_range_filter(rng: random.Random) -> Example:
    """Filter by a numeric range (between two thresholds)."""
    schema = rng.choice(_CSV_SCHEMAS)
    content, rows = _generate_csv_data(schema, rng)
    num_cols = list(schema["numeric_columns"].keys())
    num_col = rng.choice(num_cols)
    col_idx = _col_index(schema, num_col)
    output_col = schema["columns"][0]

    values = sorted(set(int(row[col_idx]) for row in rows))
    if len(values) < 3:
        return _numeric_filter(rng)
    lo_idx = rng.randint(0, len(values) // 3)
    hi_idx = rng.randint(len(values) * 2 // 3, len(values) - 1)
    lo = values[lo_idx]
    hi = values[hi_idx]
    matching = [row[0] for row in rows if lo <= int(row[col_idx]) <= hi]
    if not matching or len(matching) == len(rows):
        return _numeric_filter(rng)

    path = schema["path"]
    task = (
        f"Print the {output_col} values from {path} where "
        f"{num_col} is between {lo} and {hi} (inclusive)."
    )
    expected = "\n".join(matching) + "\n"
    files = [FileSpec(path=path, content=content)]
    return _make_example(task, files, expected, schema)


_SUB_TYPES = [
    # Easy
    _numeric_filter,
    _string_filter,
    _count_by_numeric,
    _count_by_category,
    # Medium
    _multi_condition_filter,
    _sorted_filter,
    _max_min_aggregate,
    _print_non_id_column,
    _count_multi_condition,
    # Hard
    _top_n_by_column,
    _count_distinct,
    _sum_numeric,
    _filter_two_categories,
    _numeric_range_filter,
]


def generate_csv_filtering_examples(
    n: int,
    rng: random.Random,
) -> list[Example]:
    """Generate n csv-filtering examples using the given RNG."""
    return [rng.choice(_SUB_TYPES)(rng) for _ in range(n)]
