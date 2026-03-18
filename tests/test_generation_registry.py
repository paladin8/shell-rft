"""Tests for the generation registry."""

from shell_rft.generation import GENERATORS


def test_registry_contains_all_families():
    expected = {"file_counting", "content_search", "topk_by_size", "csv_filtering"}
    assert set(GENERATORS.keys()) == expected


def test_registry_values_are_callable():
    for gen_fn in GENERATORS.values():
        assert callable(gen_fn)
