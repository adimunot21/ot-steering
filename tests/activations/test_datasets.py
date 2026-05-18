"""Tests for ot_steering.activations.datasets."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from ot_steering.activations.datasets import (
    load_refusal_pairs,
    load_sentiment_pairs,
    load_truthfulness_pairs,
)


def test_sentiment_pairs_load_and_have_50_entries() -> None:
    pairs = load_sentiment_pairs()
    assert len(pairs) == 50
    for pos, neg in pairs:
        assert pos and neg
        assert pos != neg


def test_truthfulness_pairs_load_and_have_50_entries() -> None:
    pairs = load_truthfulness_pairs()
    assert len(pairs) == 50
    for true_s, false_s in pairs:
        assert true_s and false_s
        assert true_s != false_s


def test_refusal_pairs_load_and_have_50_entries() -> None:
    pairs = load_refusal_pairs()
    assert len(pairs) == 50
    for harm, ok in pairs:
        assert harm and ok
        assert harm != ok


def test_missing_yaml_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_sentiment_pairs(tmp_path / "does_not_exist.yaml")


def test_schema_typo_in_yaml_is_caught(tmp_path: Path) -> None:
    from pydantic import ValidationError

    bad = tmp_path / "bad.yaml"
    bad.write_text(
        yaml.safe_dump(
            {
                "name": "x",
                "description": "y",
                "pairs": [{"positiv": "wrong key", "negative": "n"}],  # typo
            }
        )
    )
    with pytest.raises(ValidationError):
        load_sentiment_pairs(bad)
