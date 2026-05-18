"""Global RNG seeding for reproducibility.

This module exists so every experiment can hit a single function and trust that
*all* sources of randomness it cares about are pinned: Python's ``random``,
NumPy, PyTorch CPU/CUDA, and cuDNN's nondeterministic kernels. Without that
last bit, two GPU runs of the same script can disagree at the third decimal
place — which silently corrupts ablations.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_all_seeds(seed: int) -> None:
    """Seed every RNG that this project touches.

    Seeds Python's ``random`` module, NumPy, PyTorch (CPU + all CUDA devices),
    and configures cuDNN for deterministic kernels. Also sets the
    ``PYTHONHASHSEED`` environment variable so hash-based collections are
    deterministic across processes.

    Args:
        seed: A non-negative integer used to seed every RNG. The same value
            yields the same results across runs on the same hardware and
            library versions.

    Raises:
        ValueError: If ``seed`` is negative or not an int.
    """
    if not isinstance(seed, int):
        raise ValueError(f"seed must be int, got {type(seed).__name__}")
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Deterministic cuDNN. We accept the speed hit because reproducibility
    # of headline numbers matters more than the last 10% of throughput.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
