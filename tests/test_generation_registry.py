"""Tests for the generation registry."""

from shell_rft.generation import GENERATORS


def test_registry_contains_file_counting():
    assert "file_counting" in GENERATORS


def test_registry_values_are_callable():
    for gen_fn in GENERATORS.values():
        assert callable(gen_fn)
