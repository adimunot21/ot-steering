"""Shared pytest configuration.

The ``slow`` marker tags tests that download model weights or do non-trivial
inference. They are skipped in the default ``pytest -q`` run and need
``pytest --run-slow`` to execute.
"""

from __future__ import annotations

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="run tests marked @pytest.mark.slow (model downloads, integration tests)",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if config.getoption("--run-slow"):
        return
    skip_slow = pytest.mark.skip(reason="@pytest.mark.slow — pass --run-slow to enable")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
