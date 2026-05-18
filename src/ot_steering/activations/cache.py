"""On-disk cache for extracted activations.

Activation extraction is *expensive* (load the model, run a forward pass on
every prompt, move tensors to CPU) and *idempotent* (re-running with the
same inputs and seed produces the same tensors). The cache short-circuits
re-extraction: given a key derived from
``(model_id, dataset_id, layer, dtype, position)``, ``load_or_extract``
returns the cached tensor if present and otherwise calls the supplied
``extractor_fn`` and writes its result.

The cache lives at ``data/cache/`` (gitignored). Filenames are
``<readable-prefix>__<8-char-hash>.pt`` so a human can browse them.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable
from pathlib import Path

import torch

from ot_steering.utils.logging import get_logger

_log = get_logger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CACHE_DIR = _PROJECT_ROOT / "data" / "cache"


def _sanitise_for_filename(text: str) -> str:
    """Make ``text`` safe to embed in a filename: alnum, underscore, hyphen."""
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", text)
    return cleaned.strip("_") or "x"


def cache_key(
    model_id: str,
    dataset_id: str,
    layer: int,
    dtype: str,
    position: str,
) -> str:
    """Return a stable cache key for a ``(model, dataset, layer, dtype, position)`` tuple.

    The key is ``<readable-prefix>__<8-char-hash>``. The prefix lets a human
    skim ``ls data/cache/``; the hash provides collision safety across the
    full input tuple. The hash is over the original (non-sanitised) tuple
    so two distinct inputs that sanitise to the same prefix still differ.

    Args:
        model_id: Hugging Face model id, e.g. ``"gpt2"``.
        dataset_id: Logical dataset name, e.g. ``"sentiment_pairs_v1"``.
        layer: Transformer layer index.
        dtype: Tensor dtype name, e.g. ``"float16"``.
        position: Token position extracted, e.g. ``"last_token"`` or ``"all"``.

    Returns:
        Cache key string.
    """
    raw = "|".join([model_id, dataset_id, str(layer), dtype, position])
    digest = hashlib.blake2b(raw.encode("utf-8"), digest_size=8).hexdigest()
    prefix = _sanitise_for_filename(f"{model_id}_{dataset_id}_L{layer}_{position}")
    return f"{prefix}__{digest}"


def _path_for(cache_dir: Path, key: str) -> Path:
    return cache_dir / f"{key}.pt"


def load_or_extract(
    cache_dir: Path,
    key: str,
    extractor_fn: Callable[[], torch.Tensor],
) -> torch.Tensor:
    """Return cached tensor for ``key``, computing and writing it on a miss.

    Args:
        cache_dir: Directory holding the ``.pt`` files. Created if missing.
        key: Cache key (use :func:`cache_key`).
        extractor_fn: Zero-argument callable returning the tensor to cache
            on a miss. Called at most once.

    Returns:
        The cached (or freshly extracted) tensor.

    Raises:
        ValueError: If ``extractor_fn`` returns a non-tensor.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = _path_for(cache_dir, key)
    if path.is_file():
        _log.info("cache hit  : %s", path.name)
        cached: torch.Tensor = torch.load(path, map_location="cpu")
        return cached

    _log.info("cache miss : %s (extracting...)", path.name)
    tensor = extractor_fn()
    if not isinstance(tensor, torch.Tensor):
        raise ValueError(f"extractor_fn returned {type(tensor).__name__}, expected torch.Tensor")
    # Save to a tmp path and atomically rename so a crash mid-write doesn't
    # leave a half-written cache entry.
    tmp = path.with_suffix(".pt.tmp")
    torch.save(tensor.detach().cpu(), tmp)
    tmp.replace(path)
    _log.info(
        "cache write: %s (%s, %d MB)",
        path.name,
        tuple(tensor.shape),
        path.stat().st_size // (1024**2),
    )
    return tensor
