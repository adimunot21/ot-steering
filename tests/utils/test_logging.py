"""Tests for ot_steering.utils.logging."""

from __future__ import annotations

import logging

import pytest

from ot_steering.utils.logging import get_logger


def test_returns_logger_instance() -> None:
    log = get_logger("ot_steering.foo")
    assert isinstance(log, logging.Logger)


def test_internal_name_kept_verbatim() -> None:
    log = get_logger("ot_steering.utils.seed")
    assert log.name == "ot_steering.utils.seed"


def test_external_name_namespaced_under_project_root() -> None:
    log = get_logger("my_notebook")
    assert log.name.startswith("ot_steering.ext.")


def test_root_handler_installed_once() -> None:
    get_logger("ot_steering.a")
    get_logger("ot_steering.b")
    root = logging.getLogger("ot_steering")
    assert len(root.handlers) == 1
    assert root.propagate is False


def test_actually_emits_record() -> None:
    # The project logger has propagate=False, so neither caplog (root) nor
    # capsys (stderr) can see records — attach our own handler to verify.
    records: list[logging.LogRecord] = []

    class _List(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    listener = _List(level=logging.DEBUG)
    project_logger = logging.getLogger("ot_steering")
    project_logger.addHandler(listener)
    try:
        log = get_logger("ot_steering.emit_test")
        log.info("hello-world")
    finally:
        project_logger.removeHandler(listener)

    assert any(
        r.name == "ot_steering.emit_test" and "hello-world" in r.getMessage() for r in records
    )


def test_empty_name_raises() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        get_logger("")
