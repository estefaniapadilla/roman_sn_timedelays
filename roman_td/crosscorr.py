"""
crosscorr.py
============
[TODO] Fast GP-based cross-correlator for time-delay estimation.

Planned approach
----------------
1. Fit a Gaussian Process (Matern-3/2 kernel) independently to each image's
   light curve in each band. The GP interpolates over gaps and gives a
   smooth posterior over the flux time series.

2. Cross-correlate the GP means (or posterior samples) between image pairs
   to estimate the time delay. The lag that maximises the cross-correlation
   is a fast, unbiased estimate of Delta_t.

3. Estimate uncertainty from the spread of cross-correlations across GP
   posterior samples and/or across bands.

4. Optionally estimate the magnification ratio mu_i / mu_j from the
   flux ratio of the GP means at matched phases.

Expected speed: ~5 seconds per system (vs. 5-30 min for BayeSN nested sampling).
Expected accuracy: ~1-3 day uncertainty for Roman 5-day cadence (vs. ~0.5-1 day
for BayeSN), sufficient for population statistics and as priors for the full fit.

Planned API
-----------
from roman_td.crosscorr import gp_cross_correlate

result = gp_cross_correlate(
    tab,           # photometry table with columns (mjd, filter, flux, fluxerr, image)
    images,        # list of image labels, e.g. ["a", "b"]
    bands,         # list of filter names to use
    lag_range=(-200, 200),   # search range in days
)
# result["delays"]  : dict image -> best-fit delay (days)
# result["delay_err"]: dict image -> 1-sigma uncertainty (days)
# result["mu_ratio" ]: dict image -> flux ratio relative to reference image
"""


def gp_cross_correlate(tab, images, bands, lag_range=(-200, 200)):
    raise NotImplementedError(
        "GP cross-correlator not yet implemented. "
        "Use roman_td.sntd_wrapper.measure_one() or "
        "roman_td.bayesn_wrapper.fit_system() for now."
    )
