#!/usr/bin/env python
"""Sanity-check the local environment for the ot-steering project.

Prints versions, CUDA visibility, and runs a tiny GPU matmul to confirm that
CUDA is not only installed but actually usable. Exits 0 on success, 1 on the
first failure encountered.

Run:
    python scripts/verify_env.py
"""

from __future__ import annotations

import importlib
import platform
import sys
import traceback

# Packages to report. Key is the import name; value is the human-friendly label.
_PACKAGES: dict[str, str] = {
    "torch": "PyTorch",
    "transformers": "transformers",
    "accelerate": "accelerate",
    "bitsandbytes": "bitsandbytes",
    "ot": "POT (Python Optimal Transport)",
    "numpy": "NumPy",
    "scipy": "SciPy",
    "sklearn": "scikit-learn",
    "einops": "einops",
    "matplotlib": "matplotlib",
    "pydantic": "pydantic",
    "yaml": "PyYAML",
    "tqdm": "tqdm",
}


def _section(title: str) -> None:
    print()
    print(title)
    print("-" * len(title))


def _print_versions() -> list[str]:
    """Print versions of every required package. Returns the list of failures."""
    _section("Library versions")
    failures: list[str] = []
    for module_name, label in _PACKAGES.items():
        try:
            mod = importlib.import_module(module_name)
            version = getattr(mod, "__version__", "unknown")
            print(f"  {label:<32} {version}")
        except ImportError as e:
            failures.append(f"{label} ({module_name}): {e}")
            print(f"  {label:<32} MISSING ({e})")
    return failures


def _print_system_info() -> None:
    _section("System")
    print(f"  Python                          {platform.python_version()} ({sys.executable})")
    print(f"  Platform                        {platform.platform()}")


def _print_cuda_info_and_matmul() -> list[str]:
    """Inspect CUDA visibility and run a tiny GPU matmul. Returns failures."""
    _section("CUDA")
    failures: list[str] = []
    try:
        import torch
    except ImportError as e:  # pragma: no cover - already reported above
        failures.append(f"torch import failed: {e}")
        return failures

    cuda_available = torch.cuda.is_available()
    print(f"  torch.cuda.is_available()       {cuda_available}")
    if not cuda_available:
        failures.append("CUDA not available to PyTorch")
        return failures

    device_count = torch.cuda.device_count()
    print(f"  device count                    {device_count}")
    for i in range(device_count):
        name = torch.cuda.get_device_name(i)
        props = torch.cuda.get_device_properties(i)
        total_gb = props.total_memory / (1024**3)
        cap = f"{props.major}.{props.minor}"
        print(f"  device[{i}]                       {name}  (cc {cap}, {total_gb:.2f} GB)")
    print(f"  current device                  cuda:{torch.cuda.current_device()}")
    print(f"  torch CUDA build                {torch.version.cuda}")
    print(f"  cuDNN version                   {torch.backends.cudnn.version()}")

    _section("GPU matmul sanity check")
    try:
        a = torch.randn(256, 256, device="cuda")
        b = torch.randn(256, 256, device="cuda")
        c = a @ b
        torch.cuda.synchronize()
        # Confirm it actually computed something and lives on the GPU.
        assert c.shape == (256, 256)
        assert c.device.type == "cuda"
        assert torch.isfinite(c).all().item()
        print("  256x256 matmul on cuda:0        OK")
    except Exception as e:  # noqa: BLE001 - we want any failure to surface
        traceback.print_exc()
        failures.append(f"GPU matmul failed: {e}")
    return failures


def main() -> int:
    _print_system_info()
    failures = _print_versions()
    failures += _print_cuda_info_and_matmul()

    _section("Summary")
    if failures:
        print(f"  {len(failures)} FAILURE(S):")
        for f in failures:
            print(f"    - {f}")
        return 1
    print("  All checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
