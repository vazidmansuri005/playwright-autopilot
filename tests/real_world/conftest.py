"""Conftest for real-world tests — shared fixtures."""

import pytest


def pytest_collection_modifyitems(items):
    """Add 'real_world' marker to all tests in this directory."""
    for item in items:
        if "real_world" in str(item.fspath):
            item.add_marker(pytest.mark.real_world)
