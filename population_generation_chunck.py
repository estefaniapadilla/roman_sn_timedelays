"""
population_generation_chunck.py
===============================
Draw an EXTENSION batch of slsim lensed-SN Ia systems for the Roman HLTDS
population — in parallel worker processes, with per-chunk checkpointing.

Just run (in the sntd_bayesn env, from ~/time_delays):

    python population_generation_chunck.py

That draws 20,000 new lenses using 4 parallel workers (seeds 43-46, one
batch of 5,000 each), writing:

    data/chunks_deep_seed43/ ... seed46/     per-chunk safety checkpoints
    data/roman_lens_population_ext_deep_seed43.pkl ... seed46.pkl
    data/popgen_seed43.log ... seed46.log    per-worker logs (tail -f these)

Batch 2 (2026-07-11). The original 10k used seed 42 (never reused: the RNG
is seeded at process start, so seed 42 would regenerate the SAME lenses
and dedup would discard them). Originals are never modified — a separate
combine step writes a NEW file (original list first, extensions appended,
cross-batch dedup) and verifies the original portion against the original
pickle before the builder is allowed to use it.

numpy note: run in sntd_bayesn (numpy 1.26 — the same env that consumes
the pickle). A guard refuses to save under numpy >= 2.
"""

import argparse
import gc
import os
import pickle
import random
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# the pip-installed slsim ships no data/ directory (filter curves, SkyPy
# configs) — import it from the source checkout instead, which has them
SLSIM_CHECKOUT = "/home/epadill/slsim"
sys.path.insert(0, SLSIM_CHECKOUT)

import numpy as np

REPO = Path("/home/epadill/time_delays")
SKYPY_CONFIG = os.path.join(SLSIM_CHECKOUT, "data/SkyPy/roman-like.yml")


# ── helpers ───────────────────────────────────────────────────────────────────
def deduplicate_lenses(all_lenses):
    """Remove duplicate lenses based on key physical properties."""
    seen = set()
    unique_lenses = []
    duplicates = 0
    for lens in all_lenses:
        key = (
            round(lens.deflector_redshift, 4),
            round(lens.source_redshift_list[0], 4),
            round(float(lens.einstein_radius[0]), 4),
        )
        if key not in seen:
            seen.add(key)
            unique_lenses.append(lens)
        else:
            duplicates += 1
    print(f"Total before dedup: {len(all_lenses)}")
    print(f"Duplicates removed: {duplicates}")
    print(f"Unique lenses:      {len(unique_lenses)}")
    return unique_lenses


# ── one worker batch (runs in its own process) ────────────────────────────────
def run_batch(seed, tag, target, chunk_area, survey):
    import slsim
    from slsim.Lenses.lens_pop import LensPop
    from slsim.Pipelines.roman_speclite import configure_roman_filters
    from slsim.Pipelines.roman_speclite import filter_names
    import speclite.filters
    from astropy.cosmology import FlatLambdaCDM
    from astropy.units import Quantity
    import slsim.Sources as sources
    import slsim.Deflectors as deflectors
    import slsim.Pipelines as pipelines

    import warnings
    from scipy.integrate import IntegrationWarning
    warnings.filterwarnings("ignore", category=IntegrationWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning,
                            message=r"invalid value encountered.*")
    warnings.filterwarnings("ignore", message=r".*JWST PRD VERSION.*")
    warnings.filterwarnings("ignore", message=r".*pysiaf.*")
    warnings.filterwarnings("ignore", message=r".*Angular size is converted.*")

    if survey == "time_domain_wide":
        filters = ["F062", "F087", "F106", "F129", "F158"]
    else:
        filters = ["F087", "F106", "F129", "F158", "F184"]
    kwargs_variability = {"supernovae_lightcurve", *filters}

    np.random.seed(seed)
    random.seed(seed)

    configure_roman_filters()
    speclite.filters.load_filters(*filter_names())

    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    kwargs_deflector_cut = {"band": "F129", "band_max": 25,
                            "z_min": 0.01, "z_max": 3.0}
    kwargs_lens_cut = {"min_image_separation": 0.5, "max_image_separation": 10}
    time_range = np.linspace(-50, 500, 550)
    kwargs_sn = {
        "variability_model": "light_curve",
        "kwargs_variability": kwargs_variability,
        "sn_type": "Ia",
        "sn_absolute_mag_band": "bessellb",
        "sn_absolute_zpsys": "ab",
        "lightcurve_time": np.linspace(-50, 100, 31),
        "sn_modeldir": None,
    }

    chunk_dir = REPO / "data" / f"chunks_{tag}"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    save_path = REPO / "data" / f"roman_lens_population_ext_{tag}.pkl"

    all_lenses = []
    existing = sorted(chunk_dir.glob("chunk_*.pkl"))
    if existing:
        print(f"Found {len(existing)} existing chunks, loading...")
        for p in existing:
            with open(p, "rb") as f:
                all_lenses.extend(pickle.load(f))
        chunk_idx = len(existing)
        print(f"Recovered {len(all_lenses)} lenses "
              "(resumed runs re-seed, so repeats vs this batch's early "
              "chunks are possible — the final dedup removes them)")
    else:
        chunk_idx = 0

    def build_chunk():
        """One chunk draw. Raises on slsim's stochastic failures (e.g. an
        SN with an empty host-candidate list) — caller skips and retries."""
        deflector_sky_area = Quantity(value=5, unit="deg2")
        source_sky_area = Quantity(value=10, unit="deg2")
        host_sky_area = Quantity(value=2, unit="deg2")
        sky_area = Quantity(value=chunk_area, unit="deg2")

        print("  Building galaxy pipeline...", flush=True)
        galaxy_simulation_pipeline = pipelines.SkyPyPipeline(
            skypy_config=SKYPY_CONFIG, sky_area=deflector_sky_area,
            filters=None, cosmo=cosmo)

        print("  Done. Building host pipeline...", flush=True)
        galaxy_simulation_pipeline_host = pipelines.SkyPyPipeline(
            skypy_config=SKYPY_CONFIG, sky_area=host_sky_area,
            filters=None, cosmo=cosmo)

        print("  Done. Building deflectors...", flush=True)
        lens_galaxies = deflectors.EllipticalLensGalaxies(
            galaxy_list=galaxy_simulation_pipeline.red_galaxies,
            kwargs_cut=kwargs_deflector_cut,
            kwargs_mass2light={},
            cosmo=cosmo,
            sky_area=deflector_sky_area,
        )

        print("  Done. Building supernovae catalog...", flush=True)
        supernovae_catalog = sources.SupernovaeCatalog.SupernovaeCatalog(
            sn_type="Ia",
            band_list=filters,
            lightcurve_time=time_range,
            absolute_mag_band="bessellb",
            absolute_mag=None,
            mag_zpsys="ab",
            cosmo=cosmo,
            skypy_config=SKYPY_CONFIG,
            sky_area=source_sky_area,
            host_galaxy_candidate=galaxy_simulation_pipeline_host.blue_galaxies[0:2000],
        )
        supernovae_data = supernovae_catalog.supernovae_catalog(
            host_galaxy=True, lightcurve=False)

        print("  Done. Building source population...", flush=True)
        source_SNIa = sources.PointPlusExtendedSources(
            point_plus_extended_sources_list=supernovae_data,
            cosmo=cosmo,
            sky_area=source_sky_area,
            kwargs_cut={},
            catalog_type="skypy",
            source_size=None,
            extended_source_type="single_sersic",
            point_source_type="supernova",
            point_source_kwargs=kwargs_sn,
        )

        lens_pop = LensPop(
            deflector_population=lens_galaxies,
            source_population=source_SNIa,
            cosmo=cosmo,
            sky_area=sky_area,
        )
        print("  Done. Drawing lens population...", flush=True)
        return lens_pop.draw_population(
            kwargs_lens_cuts=kwargs_lens_cut, speed_factor=10)

    consecutive_failures = 0
    while len(all_lenses) < target:
        print(f"\n--- Chunk {chunk_idx+1} | Total so far: "
              f"{len(all_lenses)} / {target} ---", flush=True)
        try:
            chunk = build_chunk()
            consecutive_failures = 0
        except Exception as e:
            # slsim draws occasionally fail stochastically (seed-dependent),
            # e.g. IndexError in SN-host matching on an empty candidate
            # list. The RNG has advanced, so simply drawing again gives a
            # different population — skip and retry.
            consecutive_failures += 1
            print(f"chunk draw FAILED ({type(e).__name__}: {e}) — "
                  f"retrying with a fresh draw "
                  f"[{consecutive_failures}/10 consecutive]", flush=True)
            gc.collect()
            if consecutive_failures >= 10:
                raise RuntimeError(
                    "10 consecutive chunk failures — this is systematic, "
                    "not stochastic; check the config") from e
            continue

        print(f"Chunk {chunk_idx+1} yielded {len(chunk)} lenses", flush=True)
        all_lenses.extend(chunk)

        chunk_path = chunk_dir / f"chunk_{chunk_idx:04d}.pkl"
        with open(chunk_path, "wb") as f:
            pickle.dump(chunk, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Saved chunk to {chunk_path}", flush=True)

        del chunk
        gc.collect()
        chunk_idx += 1

    print(f"\nDone! Total lenses collected: {len(all_lenses)}")
    all_lenses = deduplicate_lenses(all_lenses)

    # pickle-compatibility guard: the consumer (build_training_set.py in
    # sntd_bayesn) runs numpy 1.26 — a pickle written under numpy >= 2 may
    # not unpickle there. Refuse to write one.
    assert np.__version__.startswith("1."), (
        f"numpy {np.__version__}: run this in the sntd_bayesn env "
        "(numpy 1.x) so the pickle stays loadable by the builder")

    import slsim as _slsim
    out = {
        "lens_population": all_lenses,
        "meta": {
            "created_utc": datetime.utcnow().isoformat(),
            "n_lenses": len(all_lenses),
            "n_chunks": chunk_idx,
            "chunk_sky_area_deg2": chunk_area,
            "slsim_version": getattr(_slsim, "__version__", "unknown"),
            "numpy_version": np.__version__,
            "seed_numpy": seed,
            "seed_python": seed,
            "survey": survey,
            "notes": f"extension batch seed {seed}; combine step appends "
                     "to (a copy of) the original and verifies the "
                     "original portion — originals never modified",
        },
    }
    with open(save_path, "wb") as f:
        pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved {len(all_lenses)} lenses to {save_path}")


# ── main: spawn workers and wait ─────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=20000,
                    help="TOTAL new lenses across all workers")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed_base", type=int, default=43,
                    help="first worker seed (workers use base, base+1, ...);"
                         " 42 = the original 10k batch, never reuse it")
    ap.add_argument("--chunk_area", type=float, default=1000,
                    help="deg2 of lens sky per chunk")
    ap.add_argument("--survey", default="time_domain_deep",
                    choices=["time_domain_deep", "time_domain_wide"])
    ap.add_argument("--batch_seed", type=int, default=None,
                    help=argparse.SUPPRESS)  # internal: run ONE batch inline
    args = ap.parse_args()

    if args.batch_seed is not None:   # child mode
        run_batch(args.batch_seed, f"deep_seed{args.batch_seed}",
                  args.target, args.chunk_area, args.survey)
        return

    if args.seed_base <= 42:
        sys.exit("seed_base must be > 42 (42 was the original batch)")

    per_worker = int(np.ceil(args.target / args.workers))
    seeds = [args.seed_base + i for i in range(args.workers)]
    print(f"Spawning {args.workers} workers, {per_worker} lenses each, "
          f"seeds {seeds}")
    procs = {}
    for s in seeds:
        log = REPO / "data" / f"popgen_seed{s}.log"
        cmd = [sys.executable, os.path.abspath(__file__),
               "--batch_seed", str(s), "--target", str(per_worker),
               "--chunk_area", str(args.chunk_area), "--survey", args.survey]
        procs[s] = (subprocess.Popen(cmd, stdout=open(log, "w"),
                                     stderr=subprocess.STDOUT), log)
        print(f"  seed {s} -> {log}")

    failed = []
    for s, (p, log) in procs.items():
        code = p.wait()
        status = "OK" if code == 0 else f"FAILED (exit {code})"
        print(f"worker seed {s}: {status}  (log: {log})")
        if code != 0:
            failed.append(s)

    if failed:
        sys.exit(f"workers failed: {failed} — check their logs; rerunning "
                 "the same command resumes from their chunk checkpoints")
    print("\nAll workers done. Extension pickles in data/ — next step is "
          "the combine+verify script.")


if __name__ == "__main__":
    main()
