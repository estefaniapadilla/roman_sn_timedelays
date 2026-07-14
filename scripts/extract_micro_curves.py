"""
extract_micro_curves.py
=======================
Export per-image microlensing light curves + lens truth from the
100-lens precomputed pickle (collaborator GPU maps) into portable
per-lens .npz files that the sntd_bayesn builder can read (numpy .npz is
numpy-version-portable; the source pickle is not).

Run in roman_ml (numpy 2 + astropy 8 + lenstronomy/sncosmo/skypy stack):

    PYTHONPATH=/home/epadill/slsim conda run -n roman_ml \
        python scripts/extract_micro_curves.py [--limit N]

Output: data/microlensing_data/curves_100/lens_XXX__micro.npz with
    times      (T,)            days since ~explosion (see note)
    dmag       (B, N_img, T)   micro delta-mag per band/image (0 = none)
    bands      (B,)            band names
    z_lens, z_source, arrival_days (N_img,), macro_mu (N_img,)
    kappa_star, kappa_tot, shear (N_img,)   [-1 where unavailable]

TIME ANCHOR NOTE: slsim's SN morphology clock starts at ~explosion
(photosphere radius ~ v*t). The builder aligns t=0 here to
(BayeSN peak - ~17.5*(1+z) observer days). Micro curves vary slowly
(~0.02 mag / 100 d typical), so a few-day anchor error is negligible
(<~0.002 mag).
"""

import argparse
import gzip
import os
import pickle
import sys
import time as walltime

sys.path.insert(0, "/home/epadill/slsim")

import numpy as np

from slsim.Pipelines.roman_speclite import configure_roman_filters

PICKLE = ("/home/epadill/time_delays/data/microlensing_data/"
          "lSNe_100_precomputed/precomputed_sne_lenses_100.pkl.gz")
OUTDIR = ("/home/epadill/time_delays/data/microlensing_data/curves_100")
BANDS = ["F087", "F106", "F129", "F158", "F184"]   # deep tier
TIMES = np.arange(0.0, 600.1, 5.0)                 # observer days, slow signal


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--start", type=int, default=0,
                    help="first lens index (parallel slicing)")
    ap.add_argument("--end", type=int, default=None,
                    help="one-past-last lens index")
    args = ap.parse_args()

    configure_roman_filters()
    os.makedirs(OUTDIR, exist_ok=True)

    print("loading pickle (950 MB gz)...", flush=True)
    with gzip.open(PICKLE, "rb") as f:
        lenses = pickle.load(f)
    end = args.end if args.end is not None else len(lenses)
    if args.limit:
        end = min(end, args.start + args.limit)
    indices = range(args.start, min(end, len(lenses)))
    print(f"processing lenses {indices.start}..{indices.stop - 1}", flush=True)

    n_ok = n_fail = n_skip = 0
    for i in indices:
        lens = lenses[i]
        out_path = os.path.join(OUTDIR, f"lens_{i:03d}__micro.npz")
        if os.path.exists(out_path):
            n_skip += 1
            continue
        t0 = walltime.time()
        try:
            z_l = float(lens.deflector_redshift)
            z_s = float(lens.source_redshift_list[0])
            arrival = np.array(
                [float(v) for v in np.ravel(lens.point_source_arrival_times()[0])])
            macro_mu = np.array(
                [float(v) for v in np.ravel(lens.point_source_magnification()[0])])
            n_img = len(arrival)

            dmag = np.zeros((len(BANDS), n_img, len(TIMES)), dtype=np.float32)
            for bi, band in enumerate(BANDS):
                dm = lens._point_source_magnitude_microlensing(
                    band, TIMES, source_index=0)
                dm = np.asarray(dm, dtype=float)
                if dm.shape != (n_img, len(TIMES)):
                    raise ValueError(f"unexpected micro shape {dm.shape}")
                dmag[bi] = dm

            # the SN morphology model has a finite time range; beyond it the
            # curve is NaN (SN flux is ~zero there anyway). Hold the last
            # finite value forward so the builder never sees NaN.
            finite_t = np.isfinite(dmag).all(axis=(0, 1))
            t_valid_max = float(TIMES[finite_t].max()) if finite_t.any() else 0.0
            for bi in range(dmag.shape[0]):
                for im in range(dmag.shape[1]):
                    c = dmag[bi, im]
                    bad = ~np.isfinite(c)
                    if bad.any() and (~bad).any():
                        last = np.flatnonzero(~bad)[-1]
                        c[bad] = c[last]
                    elif bad.all():
                        c[:] = 0.0

            try:
                ks, kt, sh, _ = \
                    lens._microlensing_parameters_for_image_positions_single_source(
                        BANDS[0], 0)
                ks = np.ravel(np.asarray(ks, dtype=float))[:n_img]
                kt = np.ravel(np.asarray(kt, dtype=float))[:n_img]
                sh = np.ravel(np.asarray(sh, dtype=float))[:n_img]
            except Exception:
                ks = kt = sh = np.full(n_img, -1.0)

            np.savez_compressed(
                out_path, times=TIMES, dmag=dmag,
                bands=np.array(BANDS), z_lens=z_l, z_source=z_s,
                arrival_days=arrival, macro_mu=macro_mu,
                kappa_star=ks, kappa_tot=kt, shear=sh,
                t_valid_max=t_valid_max)
            n_ok += 1
            print(f"[{i:3d}] ok  n_img={n_img}  z_s={z_s:.2f}  "
                  f"micro std max={dmag.std(axis=-1).max():.3f} mag  "
                  f"({walltime.time()-t0:.0f}s)", flush=True)
        except Exception as e:
            n_fail += 1
            print(f"[{i:3d}] FAIL {type(e).__name__}: {e}", flush=True)

    print(f"\ndone: {n_ok} ok, {n_fail} failed, {n_skip} skipped "
          f"-> {OUTDIR}", flush=True)


if __name__ == "__main__":
    main()
