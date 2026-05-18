"""Tests for ot_steering.utils.config."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from ot_steering.utils.config import BaseConfig


class _ExampleConfig(BaseConfig):
    name: str
    n_steps: int


def test_subclass_constructs_with_valid_fields() -> None:
    cfg = _ExampleConfig(name="run-1", n_steps=10)
    assert cfg.name == "run-1"
    assert cfg.n_steps == 10


def test_unknown_field_is_forbidden() -> None:
    with pytest.raises(ValidationError):
        _ExampleConfig(name="x", n_steps=1, learning_rete=0.1)  # type: ignore[call-arg]


def test_instance_is_frozen() -> None:
    cfg = _ExampleConfig(name="x", n_steps=1)
    with pytest.raises(ValidationError):
        cfg.n_steps = 2  # type: ignore[misc]


def test_type_coercion_is_strict_enough_to_catch_mistakes() -> None:
    with pytest.raises(ValidationError):
        _ExampleConfig(name="x", n_steps="not-an-int")  # type: ignore[arg-type]


def test_whitespace_is_stripped() -> None:
    cfg = _ExampleConfig(name="  padded  ", n_steps=1)
    assert cfg.name == "padded"
