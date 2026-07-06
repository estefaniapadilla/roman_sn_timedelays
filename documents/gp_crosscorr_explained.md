# The GP Cross-Correlator, Explained

A detailed guide to `roman_td/crosscorr.py` — what it does, how every step
works, what goes in, what comes out, what it costs, and how its mistakes are
contained. Companion to `pipeline_architecture.md` §2 (the specification);
this document is the *understanding*.

---

## 1. What it is, in one paragraph

`gp_cross_correlate()` estimates the time delay between the images of a
lensed supernova **without any supernova model** — no SALT, no BayeSN, no
nested sampling, no likelihood. It smooths each image's noisy light curve
into a continuous curve using a Gaussian process, then slides one smooth
curve past the other in time and finds the shift at which they line up best.
That shift is the delay. It runs in ~4–7 seconds per system versus ~2 minutes
for the SALT fit and tens of minutes for BayeSN, and it reports its own
uncertainty and a trustworthiness flag. It is the pipeline's *prior generator
and traffic router*: its job is to tell the expensive fits where to look and
which systems are hard, not to replace them.

## 2. Why it exists (role in the pipeline)

The physical fits are slow for one structural reason: a sampler proposes
thousands of whole parameter sets, and each proposal requires evaluating an
SED model at every observation. Most of those proposals are spent *finding*
the delay in a wide search window. If something cheap can shrink that window
first, the expensive fit starts near the answer and converges several times
faster.

The GP cross-correlator is that something. Three consumers use its output:

1. **Stage 3 fast mode** (`sntd_wrapper.measure_one`): today, fast mode
   re-centers its narrow ±30 d t0 window with a quick per-image
   `sncosmo.fit_lc`, which fits each image *alone* and can lock onto the
   wrong peak — and then the narrow window prevents recovery. The GP looks at
   the *relationship between the images* directly, so it places the window
   reliably, and its uncertainty says how wide the window must be.
2. **Stage 2 transformer**: `dt_gp`, its error, the flux ratio, the per-band
   scatter, and the quality flag become input features (hints).
3. **The routing rule**: `quality == "good"` → GP-primed fast fit;
   anything else → robust fit with wide bounds. The flag doubles as a
   "this system is hard" marker for all downstream analysis.

## 3. Background: what a Gaussian process is doing here

The two images are never observed at the same moments (image A on day 10,
image B on day 12), so their points cannot be compared directly. We need to
interpolate — but naive interpolation (splines, linear) treats noisy points
as exact and gives no sense of how uncertain the curve is between them.

A **Gaussian process (GP)** is a principled interpolator. It assumes the
underlying curve is smooth in a precisely-defined statistical sense, and
given the noisy points it returns, at *any* time t:

- a **mean** — the best-guess flux, and
- a **standard deviation** — how uncertain that guess is (small near data
  points, growing in gaps, largest when extrapolating beyond the data).

It can also generate **posterior draws**: entire alternative curves, each
one consistent with the data and its noise. If the data pin the curve down
tightly, all draws look alike; where the data are poor, the draws fan out.
This is what makes the delay *uncertainty* honest later — it is measured
from the data's actual constraining power, not assumed.

### 3.1 The kernel: Matérn-3/2

A GP's smoothness assumption lives in its *kernel*. We use Matérn-3/2, a
standard choice for astronomical light curves: smoother than white noise,
rougher than a Gaussian bump — it tolerates the asymmetry of a real SN light
curve (fast rise, slow decline) without ringing. Two hyperparameters, both
fitted to each light curve by maximizing the GP marginal likelihood:

- **amplitude** — how large the flux variations are;
- **length scale** — over how many days the curve varies. Bounded to
  **[3, 40] days**: the lower bound stops the GP from tracking noise
  point-to-point (which would let it "explain" anything and correlate with
  everything); the upper bound stops it over-smoothing the peak (which
  carries the timing information).

### 3.2 Choices that matter (and why)

- **Per-point noise**: each observation's own `fluxerr` enters as that
  point's noise level (`alpha` in sklearn), so a low-SNR point constrains
  the curve less than a high-SNR one. Nothing is assumed about the noise
  beyond what the survey depth already told us.
- **Zero-mean prior** (`normalize_y=False`): away from data the GP mean
  decays to zero — physically correct, since zero flux *is* the truth before
  the SN rises and after it fades. Normalizing to the data mean would invent
  a phantom baseline flux in the gaps.
- **One GP per (image, band)**: bands are never mixed in a single GP; the SN
  has genuinely different light-curve shapes in different bands, and keeping
  them separate is also what later allows a per-band delay comparison
  (chromatic disagreement is a future microlensing indicator).
- **Validity mask**: each GP is trusted only within one length scale of a
  real observation. Outside that it is extrapolating toward the prior and
  must not participate in any comparison.
- **Draws by Cholesky, not `sample_y`**: sklearn's built-in sampler routes
  through numpy's `multivariate_normal`, which does an internal SVD — ~100×
  slower than a Cholesky factorization for the same result. This one change
  took the per-system cost from ~23 s to ~4 s.

## 4. The algorithm, step by step

All steps live in `gp_cross_correlate()` and its helpers; constants at the
top of `crosscorr.py`.

**Step 0 — setup.** Build a uniform evaluation grid over the full observed
time span at `GRID_DT = 0.5` day spacing, and the trial-lag grid over
`lag_range` (default ±200 d) at the same spacing.

**Step 1 — GP per (image, band)** (`_fit_gp_band`). Skip any (image, band)
with fewer than 5 points. Normalize flux by its max for numerical stability,
fit the two kernel hyperparameters, evaluate mean + covariance on the grid,
Cholesky-sample `n_draws = 30` posterior curves, and record the validity
mask.

**Step 2 — choose the reference image.** The image with the brightest GP
peak in any band. Brightest = highest SNR = most reliable clock. (Same
convention as the BayeSN fit's `fit_system`, so delays are comparable.) The
reference's peak time `t_peak_ref` is also reported — Stage 2 uses it as the
tokenization zero-point.

**Step 3 — bright masks.** For each curve, mark grid points where the GP
mean exceeds `BRIGHT_FRAC = 30%` of its peak (within the validity mask).
This is the fix for the subtlest failure mode we hit: **normalized
correlation of two flat zero-flux baselines is spuriously high** (two
near-constant noisy segments always "agree"), which created fake correlation
peaks at extreme lags — one test run confidently reported −179 d for a true
+25 d delay. Requiring the bright parts of *both* curves to overlap
(≥ `MIN_BRIGHT_OVERLAP` = 5 grid points = 2.5 d) kills these outright, and
as a bonus lets most trial lags be skipped entirely (speed).

**Step 4 — correlation vs lag** (`_xcorr_curve`). For each shared band and
each surviving trial lag τ: overlay A(t) against B(t + τ), restrict to grid
points where both GPs are valid (≥ `MIN_OVERLAP_PTS` = 20 points = 10 d),
and compute the **inverse-variance-weighted Pearson correlation** — points
where either GP is uncertain contribute less (weight `1/(var_A + var_B)`).
The result is a curve C(τ): how well the images line up at every trial
shift.

Sign convention: **positive τ means the image arrives later than the
reference** — matching the sign of the true delays
(`arrival_time_image − arrival_time_reference`) everywhere in the repo.

**Step 5 — the point estimate.** Average C(τ) across bands, take the argmax,
and refine it by parabolic interpolation through the three points around the
maximum (`_refine_peak`) — this recovers sub-grid precision, so the 0.5-day
lag step does not quantize the answer.

**Step 6 — the uncertainty.** Repeat step 4–5 for each of the 30 posterior
draws in each band (only at lags the mean curve accepted). Each draw is a
"plausible alternative reality" of the data; the scatter of its delay
answers is the delay uncertainty. We use a **robust scatter**
(1.4826 × median absolute deviation) so a few crazy draws cannot inflate the
error bar. This is `dt_gp_err`.

**Step 7 — the flux ratio.** Per band: (GP peak of the image) ÷ (GP peak of
the reference), with an uncertainty from the same posterior draws; then an
inverse-variance-weighted average across bands. This is a proxy for the
**total magnification ratio** (macro × micro — the GP cannot tell them
apart; with no microlensing in the current sims it equals the macro ratio).
Note it needs no temporal overlap at all, which is why it stays accurate
even on systems whose *delay* is unmeasurable.

**Step 8 — the quality flag** (`_classify`):

| flag | condition | meaning / routing |
|---|---|---|
| `good` | single correlation peak and `dt_gp_err` ≤ 5 d | trust it; prime fast mode |
| `broad` | single peak but error > 5 d or undefined | usable hint; run robust |
| `multipeak` | a rival peak ≥ 80% of the primary, > 10 d away | aliasing; run robust |
| `fail` | max correlation < 0.3, or no valid overlap | nothing to correlate; run robust |

Deliberately **not** based on the correlation peak's width: smooth SN light
curves always produce a wide C(τ) peak, so width measures the light curve's
shape, not the estimate's precision (an early version flagged everything
"broad" for exactly this reason). The draw scatter is the honest precision.

## 5. Inputs and outputs (exact)

### Inputs

```python
gp_cross_correlate(tab, images, bands=None, lag_range=(-200, 200),
                   n_draws=30, rng=None)
```

| argument | content |
|---|---|
| `tab` | canonical photometry table (§6.1 of the architecture doc): one row per (image, epoch, band) with columns `mjd, filter, flux, fluxerr, zp, zpsys, image` (legacy `time`/`band` names also accepted) |
| `images` | image labels present in the table, ≥ 2 |
| `bands` | filters to use; `None` = all in the table; only bands with ≥ 5 points for *both* images of a pair contribute |
| `lag_range` | search window in days; must exceed the population's largest measurable delay |
| `n_draws` | posterior draws for the uncertainty (30 is plenty; cost is linear in it) |
| `rng` | `np.random.Generator` for reproducibility |

Notably absent: **no redshifts, no SN template, no lens model, no cosmology.**
Pure signal processing. (The redshifts the survey guarantees are used
*downstream* — Stage 2 tokenization and Stage 3 fits — not here.)

### Output (architecture doc §6.3)

```python
{
  "ref_image": "image_1",          # brightest image = delay zero-point
  "t_peak_ref": 60012.3,           # its GP peak time (MJD)
  "wall_time_s": 4.2,
  "per_image": {
    "image_2": {
      "dt_gp": 22.1,               # delay vs ref, days; + = arrives later
      "dt_gp_err": 1.7,            # robust scatter of GP-draw estimates
      "dt_per_band": {"f129": 21.8, "f158": 22.5},   # chromatic diagnostic
      "flux_ratio": 0.51,          # total magnification-ratio proxy
      "flux_ratio_err": 0.03,
      "flux_ratio_per_band": {...},
      "quality": "good",           # good | broad | multipeak | fail
    },
  },
}
```

The population driver (`scripts/run_gp_population.py`) flattens this into
`gp_benchmark.ecsv` (one row per non-reference image) together with the
truth, residuals, image SNRs and redshifts — same seeding as the SNTD
driver, so rows are comparable lens-by-lens with `delay_benchmark_*.ecsv`.

## 6. Computational cost

**~4–7 s per system on one core**, deterministic (no sampler, nothing to
converge). Where it goes:

| piece | cost | scaling |
|---|---|---|
| GP hyperparameter fits | ~0.1 s each × (images × bands) | cubic in points per curve (~70 → trivial) |
| GP grid evaluation + Cholesky draws | ~0.2 s each | cubic in grid length (~800–1500 points) |
| correlation: mean curves | small | (accepted lags) × (overlap length) |
| correlation: 30 draws × bands | the bulk of the remainder | linear in draws; only at lags the mean accepted |

Population math: ~half of attempted lenses yield a usable ≥2-image system,
so 1000 successes ≈ 2000 attempts ≈ 4.5 core-hours ≈ **~10 min on 32
workers** (idle machine). Memory is negligible. Use
`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1` when running many joblib workers so
BLAS doesn't oversubscribe cores.

For contrast: the same 1000 systems cost hours via the SALT fast fit and
days via BayeSN. The GP is ~30× cheaper than the cheapest physical fit.

## 7. Validation so far

**Synthetic tests** (known truth, SN-like asymmetric pulse, 2 bands,
peak SNR 20, 5 d cadence, true Δt = +25 d, true ratio 0.5; 8 noise
realizations): median delay residual −0.3 d, scatter 0.7 d, reported errors
~1 d (consistent with the actual scatter), flux ratio recovered to ~4%, all
realizations flagged `good`. A near-zero-delay system (+2 d) is recovered
(+2.3 d). An image replaced by pure noise is *not* flagged `good`.

**First 10 real slsim systems** (Roman deep survey sim, ~7 s each):

- 6 × `good`: every one within 0.2–3 d of truth, honest error bars;
- 1 × `multipeak`: the estimate was wrong (aliased) — and the flag said so;
- 3 × `fail`: true delays of 245–339 d, where the SN genuinely fades before
  the second image rises — nothing to correlate, correctly refused;
- flux ratios accurate on *all ten*, including the delay-failures
  (e.g. 0.505 measured vs 0.505 true).

The load-bearing property: **every wrong delay was flagged, every `good`
was actually good.**

## 8. Containment: why a fast wrong answer cannot poison the pipeline

The legitimate worry with a fast estimator is silent failure — a wrong
number that downstream stages inherit. Four containment layers:

1. **The flag gates everything.** Only `quality == "good"` estimates prime
   the fast fit; `broad`/`multipeak`/`fail` systems go to the robust fit,
   which uses wide data-driven bounds and never sees the GP number.
2. **The GP never replaces a measurement.** It sets *search windows and
   priors* for a physical fit that still sees the raw photometry. A primed
   window is generous — `dt_gp ± max(4·σ, 10 d)` — so even a
   few-days-off estimate still contains the truth.
3. **Everything is recorded.** `gp_benchmark.ecsv` keeps truth, estimate,
   error, flag, SNRs per system. Any misbehaving subpopulation (e.g. flags
   that are overconfident in some SNR range) is visible in one plot, and the
   affected systems can be re-run robustly — recovery is a re-run, not an
   archaeology dig.
4. **Stage 2 is trained not to over-trust it.** The transformer receives GP
   features with dropout (§3.3 of the architecture doc), so it learns to
   solve the problem from the light curves alone and treat the hint as a
   refinement.

The residual risk worth respecting: flag *calibration* on the real
population (is `good` reliable at every SNR and delay?). That is exactly
what the 1000-system benchmark run measures before the GP is wired into
anything.

## 9. Knobs (all constants at the top of `crosscorr.py`)

| constant | value | what it controls | when to touch it |
|---|---|---|---|
| `LS_BOUNDS` | (3, 40) d | GP smoothness limits | different cadence or transient class |
| `GRID_DT` | 0.5 d | grid + lag resolution | rarely; cost ∝ 1/GRID_DT² |
| `MIN_OVERLAP_PTS` | 20 (10 d) | min valid overlap per lag | shorter light curves |
| `BRIGHT_FRAC` | 0.30 | "bright" threshold for gating | very asymmetric image pairs |
| `MIN_BRIGHT_OVERLAP` | 5 (2.5 d) | min bright overlap per lag | (with BRIGHT_FRAC) |
| `MIN_CORR` | 0.3 | below this, `fail` | noisier surveys |
| `n_draws` | 30 | error-bar fidelity | 50+ for publication-grade σ |
| good/broad threshold | 5 d | flag boundary | tie to cadence if it changes |
