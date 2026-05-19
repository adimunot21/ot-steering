#!/usr/bin/env python
"""Regenerate every chapter figure across the project.

Calls each phase's ``experiments/make_figures.py`` in chapter order. For
phases whose figures depend on a saved run artefact (Phase 6, Phase 7),
the per-phase script reads the latest ``outputs/<run_id>/*.json`` if
present — re-run the corresponding ``run_*.py`` / ``sweep.py`` first if
you want fresh numbers.

Run:
    python scripts/make_all_figures.py
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]

# In chapter order. Phase 0 has no figures; Phase 8 is the synthesis chapter
# and reuses figures from earlier phases.
PHASE_FIGURES_SCRIPTS: tuple[Path, ...] = (
    _PROJECT_ROOT / "phases" / "phase_01_ot_foundations" / "experiments" / "make_figures.py",
    _PROJECT_ROOT / "phases" / "phase_02_gromov_wasserstein" / "experiments" / "make_figures.py",
    _PROJECT_ROOT
    / "phases"
    / "phase_03_llms_and_steering_baselines"
    / "experiments"
    / "make_figures.py",
    _PROJECT_ROOT
    / "phases"
    / "phase_04_intra_model_ot_steering"
    / "experiments"
    / "make_figures.py",
    _PROJECT_ROOT / "phases" / "phase_05_cross_model_gw" / "experiments" / "make_figures.py",
    _PROJECT_ROOT / "phases" / "phase_06_steering_transport" / "experiments" / "make_figures.py",
    _PROJECT_ROOT / "phases" / "phase_07_diagnostics" / "experiments" / "analyse_correlations.py",
)


def _run_one(script: Path) -> tuple[Path, int, float]:
    """Execute one figure-generation script. Returns (path, return_code, elapsed_seconds)."""
    if not script.is_file():
        print(f"  (skip) {script.relative_to(_PROJECT_ROOT)} — file missing")
        return script, -1, 0.0
    print(f"--- running {script.relative_to(_PROJECT_ROOT)} ---")
    t0 = time.perf_counter()
    proc = subprocess.run(
        [sys.executable, str(script)],
        cwd=_PROJECT_ROOT,
        check=False,
    )
    elapsed = time.perf_counter() - t0
    print(f"    exit={proc.returncode}  elapsed={elapsed:.1f}s\n")
    return script, proc.returncode, elapsed


def main() -> int:
    failures: list[Path] = []
    skipped: list[Path] = []
    total_elapsed = 0.0
    for script in PHASE_FIGURES_SCRIPTS:
        path, rc, elapsed = _run_one(script)
        total_elapsed += elapsed
        if rc == -1:
            skipped.append(path)
        elif rc != 0:
            failures.append(path)

    print("=" * 60)
    print(f"figure regeneration complete in {total_elapsed:.1f}s total")
    if skipped:
        print(f"  skipped ({len(skipped)}):")
        for p in skipped:
            print(f"    - {p.relative_to(_PROJECT_ROOT)}")
    if failures:
        print(f"  FAILED ({len(failures)}):")
        for p in failures:
            print(f"    - {p.relative_to(_PROJECT_ROOT)}")
        return 1
    print("  all phases OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
