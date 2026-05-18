"""Contrastive-pair datasets for steering experiments.

Each dataset is a list of ``(label_A, label_B)`` prompt pairs where the two
elements differ along the concept axis we want to steer (positive vs.
negative sentiment; true vs. false statement; harmful vs. harmless request)
and otherwise share topic and length as much as possible.

The pairs themselves live in ``configs/datasets/*.yaml`` and are loaded
through a pydantic model that makes typos or schema drift fail loudly.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import Field

from ot_steering.utils.config import BaseConfig
from ot_steering.utils.logging import get_logger

_log = get_logger(__name__)

_CONFIGS_DIR = Path(__file__).resolve().parents[3] / "configs" / "datasets"


class SentimentPair(BaseConfig):
    """A single (positive, negative) sentiment pair."""

    positive: str
    negative: str


class SentimentDatasetConfig(BaseConfig):
    """Schema for ``configs/datasets/sentiment_pairs.yaml``."""

    name: str
    description: str
    pairs: list[SentimentPair] = Field(min_length=1)


class TruthPair(BaseConfig):
    """A single (true, false) factual-statement pair."""

    true_statement: str
    false_statement: str


class TruthDatasetConfig(BaseConfig):
    """Schema for ``configs/datasets/truth_pairs.yaml``."""

    name: str
    description: str
    pairs: list[TruthPair] = Field(min_length=1)


class RefusalPair(BaseConfig):
    """A single (harmful_request, harmless_request) pair."""

    harmful_request: str
    harmless_request: str


class RefusalDatasetConfig(BaseConfig):
    """Schema for ``configs/datasets/refusal_pairs.yaml``."""

    name: str
    description: str
    pairs: list[RefusalPair] = Field(min_length=1)


def _load_yaml(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"dataset YAML not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"{path} did not parse to a mapping; got {type(data).__name__}")
    return data


def load_sentiment_pairs(path: Path | None = None) -> list[tuple[str, str]]:
    """Load and validate the sentiment dataset.

    Args:
        path: Optional override of the YAML path. Defaults to
            ``configs/datasets/sentiment_pairs.yaml``.

    Returns:
        List of ``(positive, negative)`` prompt tuples.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        pydantic.ValidationError: If the YAML schema is wrong.
    """
    cfg = SentimentDatasetConfig.model_validate(
        _load_yaml(path or _CONFIGS_DIR / "sentiment_pairs.yaml")
    )
    _log.info("loaded sentiment dataset '%s' with %d pairs", cfg.name, len(cfg.pairs))
    return [(p.positive, p.negative) for p in cfg.pairs]


def load_truthfulness_pairs(path: Path | None = None) -> list[tuple[str, str]]:
    """Load and validate the truthfulness dataset.

    Returns:
        List of ``(true_statement, false_statement)`` tuples.
    """
    cfg = TruthDatasetConfig.model_validate(_load_yaml(path or _CONFIGS_DIR / "truth_pairs.yaml"))
    _log.info("loaded truth dataset '%s' with %d pairs", cfg.name, len(cfg.pairs))
    return [(p.true_statement, p.false_statement) for p in cfg.pairs]


def load_refusal_pairs(path: Path | None = None) -> list[tuple[str, str]]:
    """Load and validate the refusal dataset.

    Returns:
        List of ``(harmful_request, harmless_request)`` tuples.
    """
    cfg = RefusalDatasetConfig.model_validate(
        _load_yaml(path or _CONFIGS_DIR / "refusal_pairs.yaml")
    )
    _log.info("loaded refusal dataset '%s' with %d pairs", cfg.name, len(cfg.pairs))
    return [(p.harmful_request, p.harmless_request) for p in cfg.pairs]
