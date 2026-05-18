"""Tests for ot_steering.eval.steering_eval.

The lexicon and refusal judges are fast pure-Python; the perplexity test is
marked ``slow`` since it loads a real model.
"""

from __future__ import annotations

from ot_steering.eval.steering_eval import (
    _refusal_judge,
    _sentiment_judge,
)


def test_sentiment_judge_positive() -> None:
    assert _sentiment_judge("That was a wonderful, perfect evening — best ever.") == 1


def test_sentiment_judge_negative() -> None:
    assert _sentiment_judge("That was a terrible, awful evening — the worst.") == -1


def test_sentiment_judge_neutral() -> None:
    assert _sentiment_judge("The carpet is beige.") == 0


def test_sentiment_judge_mixed_returns_dominant() -> None:
    # Two positives, one negative → positive wins.
    assert _sentiment_judge("Wonderful, lovely, but ultimately awful.") == 1


def test_refusal_judge_detects_canonical_refusals() -> None:
    assert _refusal_judge("I'm sorry, but I can't help with that.")
    assert _refusal_judge("As an AI, I cannot answer this.")
    assert _refusal_judge("I won't be able to do that.")


def test_refusal_judge_does_not_flag_normal_text() -> None:
    assert not _refusal_judge("Here is a recipe for chocolate cake: …")
    assert not _refusal_judge("The capital of France is Paris.")
