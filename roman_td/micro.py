"""Reader for precomputed microlensing curves (.npz).

Files are written by scripts/extract_micro_curves.py, one per lens:
    times    (T,)          observer days since ~explosion (slsim SN clock)
    dmag     (B, N_img, T) micro delta-mag per band/image (native slsim order)
    bands, z_lens, z_source, arrival_days, macro_mu, kappa_star, kappa_tot,
    shear, t_valid_max

Images here are re-sorted by arrival time at load, matching lens_truth():
image index 0 = first-arriving image, so an index into this reader is the
same index simulate_photometry() uses.
"""

from typing import Any, Dict

import numpy as np

# BayeSN/SALT-like SNe Ia rise in ~17.5 rest-frame days; used to convert
# "days from peak" (builder clock) to "days since explosion" (micro clock).
REST_DAYS_EXPLOSION_TO_PEAK = 17.5


class MicroCurves:
    """Per-image microlensing delta-mag curves for one lens system."""

    def __init__(self, npz_path: str):
        d = np.load(npz_path)
        self.path = str(npz_path)
        self.times = np.asarray(d["times"], dtype=float)
        arrival = np.asarray(d["arrival_days"], dtype=float)
        order = np.argsort(arrival)
        self.arrival_days = arrival[order]
        self.dmag = np.asarray(d["dmag"], dtype=float)[:, order, :]
        self.bands = [str(b) for b in d["bands"]]
        self._band_ix = {b: i for i, b in enumerate(self.bands)}
        self.macro_mu = np.abs(np.asarray(d["macro_mu"], dtype=float))[order]
        self.z_lens = float(d["z_lens"])
        self.z_source = float(d["z_source"])
        self.kappa_star = np.asarray(d["kappa_star"], dtype=float)[order]
        self.kappa_tot = np.asarray(d["kappa_tot"], dtype=float)[order]
        self.shear = np.asarray(d["shear"], dtype=float)[order]
        self.t_valid_max = float(d["t_valid_max"])
        self.n_images = len(self.arrival_days)

    def truth(self) -> Dict[str, Any]:
        """lens_truth()-format dict for the external-population path."""
        return {
            "z_source": self.z_source,
            "z_lens": self.z_lens,
            "delays": self.arrival_days - self.arrival_days[0],
            "mu": self.macro_mu.copy(),
            "n_images": self.n_images,
        }

    def __call__(self, band: str, image_index: int,
                 t_since_explosion: np.ndarray) -> np.ndarray:
        """Micro delta-mag for one image, interpolated (edge-clamped)."""
        dm = self.dmag[self._band_ix[band], image_index]
        return np.interp(t_since_explosion, self.times, dm)
