"""
paths.py
========
Single source of truth for where pipeline inputs/outputs live, so results
stop accumulating in scripts/.

Layout:
    data/                inputs (slsim pickles)
    outputs/benchmarks/  results tables (*.ecsv)
    outputs/<run dirs>/  per-run payloads (e.g. bayesn_run)
    figures/             plots

find_benchmark() also checks the legacy location (scripts/) so nothing
breaks while long runs that predate this layout are still writing there.
"""

import os

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(REPO_ROOT, "data")
BENCHMARKS_DIR = os.path.join(REPO_ROOT, "outputs", "benchmarks")
TRAINING_DIR = os.path.join(REPO_ROOT, "outputs", "training")
FIGURES_DIR = os.path.join(REPO_ROOT, "figures")
_LEGACY_DIR = os.path.join(REPO_ROOT, "scripts")


def benchmark_path(name: str) -> str:
    """Where a benchmark file SHOULD be written (creates the dir)."""
    os.makedirs(BENCHMARKS_DIR, exist_ok=True)
    return os.path.join(BENCHMARKS_DIR, name)


def find_benchmark(name: str) -> str:
    """Locate an existing benchmark: outputs/benchmarks first, scripts/
    as legacy fallback. Raises FileNotFoundError with both paths shown."""
    for d in (BENCHMARKS_DIR, _LEGACY_DIR):
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        f"{name} not found in {BENCHMARKS_DIR} or {_LEGACY_DIR}")
