# Lensed SN Ia Time-Delay Pipeline — Architecture Specification Fable

Companion to `lensed_sn_pipeline_design.md` (the *why*). This document is the
*how*: for every stage, the exact inputs, the treatment applied to them, the
outputs, and the contract by which each stage feeds the next. Where a component
already exists in this repository, the real module/function is named; where it
does not, the section is marked **[TO BUILD]** and specified precisely enough
to implement without further design decisions.

Repository layout assumed throughout:

```
time_delays/
├── data/                       # slsim population pickles
├── roman_td/                   # reusable library code
│   ├── sntd_wrapper.py         # Stage 3 (SALT path)   — EXISTS
│   ├── bayesn_wrapper.py       # Stage 3 (BayeSN path) — EXISTS
│   ├── simulate.py             # Stage 0 (BayeSN sim)  — EXISTS
│   ├── crosscorr.py            # Stage 1 (GP)          — STUB, TO BUILD
│   ├── bayesncosmo.py          # BayeSN sncosmo source — EXISTS
│   ├── tokenize.py             # Stage 2 input prep    — TO BUILD
│   └── transformer.py          # Stage 2 model         — TO BUILD
├── scripts/                    # population drivers
│   ├── run_sntd_population.py      # EXISTS
│   ├── run_bayesn_population.py    # EXISTS
│   ├── run_gp_population.py        # TO BUILD
│   ├── build_training_set.py       # TO BUILD
│   └── train_transformer.py        # TO BUILD
└── documents/
```

---

## 0. The pipeline at a glance

```
                         ┌─────────────────────────────────────────────┐
                         │ Stage 0 · SIMULATION                        │
                         │ slsim pickle → photometry + truth labels    │
                         └──────┬──────────────────────────┬───────────┘
                                │ photometry table         │ truth labels
                                ▼                          │ (training only)
                         ┌─────────────────┐               │
                         │ Stage 1 · GP    │               │
                         │ cross-correlate │               │
                         └──────┬──────────┘               │
             coarse Δt ± σ,     │                          │
             flux ratio,        │                          │
             quality flag       │                          │
                                ▼                          ▼
                         ┌──────────────────────────────────────┐
                         │ Stage 2 · TRANSFORMER                │
                         │ tokens + GP hints → refined Δt, μ,   │
                         │ microlensing, SN params (each ± σ)   │
                         └──────┬───────────────────────────────┘
             tight t0/dt bounds │
             + parameter priors ▼
                         ┌──────────────────────────────────────┐
                         │ Stage 3 · PHYSICAL FIT (SNTD)        │
                         │ SALT fast/robust  or  BayeSN 2-stage │
                         └──────┬───────────────────────────────┘
                                ▼
                         benchmark table + posteriors
                         (publication-grade Δt, μ ratios)
```

Every arrow is a **data contract** defined in §6. The stages are deliberately
decoupled: Stage 3 runs today without Stages 1–2 (that is the current state of
the repo); Stage 1 improves Stage 3 without any ML; Stage 2 is the accelerant
added last.

---

## 1. Stage 0 — Simulation (photometry + truth generation)

### 1.1 Purpose

Produce, per lens system: (a) a realistic multi-image, multi-band Roman
photometry table, and (b) the ground-truth labels (delays, magnifications,
microlensing curves, SN parameters). The photometry feeds Stages 1–3; the
labels feed transformer training (Stage 2) and every benchmark.

### 1.2 Inputs

| Input | Type / location | Notes |
|---|---|---|
| Lens population | `data/roman_*_lens_population_compat.pkl` — dict with key `"lens_population"`, a list of slsim lens objects | Each lens object provides `source_redshift_list`, `deflector_redshift`, `point_source_arrival_times()`, `point_source_magnification()`, `point_source_magnitude(band, lensed, time)` |
| Survey config | `SURVEY_BANDS` in `roman_td/simulate.py` | `time_domain_deep`: F087–F184; `time_domain_wide`: F062–F158 |
| Depths | `DEFAULT_DEPTH_5SIG` (per-band 5σ AB mag). Two variants exist: `simulate.py` (Hounsell+2018 wide-tier scaling) and `sntd_wrapper.py` (lenstronomy deep-tier). **Use the one matching the survey tier and record which was used in the output.** |
| Cadence | days between visits (default 5.0) |
| RNG seed | `np.random.default_rng(seed + lens_index)` — per-system seeding so runs are reproducible and workers independent |

### 1.3 Treatment — two existing simulators, one planned extension

**Path A (SALT pipeline sim): `sntd_wrapper.extract_light_curves()`**
1. Query `lens.point_source_arrival_times()` → true arrival time per image.
2. For each band, call `lens.point_source_magnitude(band, lensed=True,
   time=obs_times)` — lenstronomy ray-tracing gives the *lensed* magnitude of
   each image at each epoch. This call dominates runtime (~80 calls per band
   at 5-day cadence over 400 days).
3. Convert mags to flux at `ZP = 25.0` (AB), add Gaussian noise with
   σ = flux(depth_5σ)/5 per band.
4. Drop non-finite epochs; keep an image only if it has data; record its peak
   SNR.

**Path B (BayeSN sim): `simulate.simulate_photometry()`**
1. Take `truth = lens_truth(lens)` (redshifts, delays, macro-μ per image).
2. Draw one BayeSN realization: `theta ~ N(0,1)`, `hostebv ~ min(Exp(0.1), 1)`,
   `hostr_v = 3.1`; scale amplitude to the correct luminosity distance via
   `set_amplitude_for_distance()` (peak M_B = SIM_MB, FlatLambdaCDM H0=70).
3. For each image k: evaluate `mu_k * model.bandflux(band, t − dt_k)` on the
   cadence grid, add Gaussian noise from the depth map.
4. **Detection cut:** an image is kept only if peak SNR ≥ 5 *and* it has ≥ 5
   observations. A system is kept only if ≥ 2 images survive.
5. **Delay re-referencing:** kept delays are re-zeroed to the first *kept*
   image (`kept_delays −= kept_delays[0]`) — the reference image after cuts is
   not necessarily slsim's image 1. Every downstream consumer must use the
   `images` list from `sim_info["truth"]`, never assume `image_1`.

**Path C [TO BUILD]: microlensing + tier injection.** Neither existing path
simulates microlensing (Path A's docstring states this explicitly — each image
gets a *flat* macro magnification). The extension:

1. For each image, draw a microlensing magnification map parameterized by the
   local convergence κ, shear γ, and stellar mass fraction s at the image
   position (available from the slsim macro-model; GERLUMPH-style maps or
   `microlensing`-package equivalents).
2. Convolve the map with the SN photosphere profile as a function of
   wavelength and phase (photosphere radius grows ~v·t with v ≈ 10⁴ km/s;
   effective radius is smaller in the blue). This produces, per image and per
   band, a smooth time-varying chromatic magnification curve `μ_micro(t, band)`.
3. Multiply into the flux *before* adding noise:
   `f = μ_macro,k · μ_micro,k(t, band) · f_SN(t − dt_k, band)`.
4. Store `μ_micro,k(t, band)` sampled on the observation grid as a truth label.

**Tier structure** (one flag per system, drawn per-system so all tiers share
the same lens population):

| Tier | Effects present | Truth labels set to |
|---|---|---|
| 0 | SN only (μ = 1, Δt as given) | micro amplitude = 0 |
| 1 | SN + macro | micro amplitude = 0 |
| 2 | SN + macro + micro | true injected curves |
| 3 | SN + macro + micro + milli | + subhalo perturbation (research-grade) |

One network trains on all tiers; the tier flag is stored for curriculum
scheduling and ablation analysis, **not** used as a network input.

### 1.4 Outputs (the Stage-0 → downstream contract)

Two artifacts per system, written by `scripts/build_training_set.py` [TO BUILD]:

**(a) Photometry table** — canonical schema (§6.1). Note the two existing
paths currently emit *different* schemas (Path A: per-image `OrderedDict` of
tables with column `time`/`band`; Path B: one combined table with
`mjd`/`filter`/`image`). §6.1 defines the canonical combined form; write thin
adapters rather than changing the existing functions.

**(b) Truth record** — one JSON per system (§6.2): redshifts, per-image
delays/macro-μ re-referenced after detection cuts, drawn SN parameters,
injected microlensing curves, tier flag, and the exact simulation config
(depths used, cadence, seed) so any system is reproducible from its record.

### 1.5 How it feeds the next stage

- **Photometry table → Stage 1** directly (the GP needs nothing else).
- **Photometry table + truth → Stage 2 training** (tokens from the table,
  regression targets from the truth record).
- **Truth → every benchmark** (Stages 1, 2, 3 all report residuals against it).

---

## 2. Stage 1 — GP cross-correlation (coarse Δt, no SN model)

### 2.1 Purpose

A model-free delay estimator running in seconds: interpolate each image's
light curve with a Gaussian process, cross-correlate image pairs, report a
coarse Δt with uncertainty and a flux ratio. Its job is *not* accuracy — it is
to place Stage 3's narrow search window reliably and to give Stage 2 a hint
feature.

### 2.2 Status

`roman_td/crosscorr.py` exists as a stub with the planned API already declared:

```python
result = gp_cross_correlate(tab, images, bands, lag_range=(-200, 200))
```

**[TO BUILD]** — implement to this spec.

### 2.3 Inputs

| Input | Source | Requirement |
|---|---|---|
| `tab` | Stage 0 canonical photometry table (§6.1) | ≥ 5 points per image in at least one shared band |
| `images` | image labels present in `tab` | ≥ 2 |
| `bands` | filters to use | only bands with data for *both* images of a pair contribute |
| `lag_range` | search window, days | default (−200, +200); must exceed the population's max delay (current Stage-3 cut: `max_delay_days=150`) |

### 2.4 Treatment (exact algorithm)

Per image pair (reference = brightest image, matching `fit_system()`'s
convention), per shared band:

1. **GP fit per image per band.** Matérn-3/2 kernel on (t, flux) with the
   photometric `fluxerr` as per-point white noise. Fit only the two kernel
   hyperparameters (amplitude, length scale) by marginal-likelihood
   maximization; bound the length scale to [3, 40] days so the GP cannot
   collapse to noise-tracking or over-smooth the peak.
2. **Evaluate** both GP means (and variances) on a common uniform grid
   (0.5-day spacing) covering the union of both images' time spans.
3. **Cross-correlate:** for each trial lag τ on the grid, compute the
   inverse-variance-weighted correlation of GP_A(t) with GP_B(t − τ) over the
   overlap region, using only grid points where *both* GP variances are below
   a cutoff (i.e., don't correlate extrapolated regions). Record the
   correlation curve C(τ).
4. **Point estimate:** τ̂ = argmax C(τ), refined by parabolic interpolation of
   the three grid points around the maximum.
5. **Uncertainty:** draw N = 50 posterior sample curves from each GP, repeat
   steps 3–4 per draw *and* per band; σ(Δt) = the standard deviation of τ̂
   across draws × bands. The **per-band scatter is kept separately** — large
   band-to-band disagreement is itself a chromatic (microlensing) indicator
   and feeds the Stage-2 features.
6. **Quality flag:** classify C(τ) as
   - `good` — single peak and σ(Δt) ≤ 5 d (one cadence step)
   - `broad` — single peak but σ(Δt) > 5 d or undefined
   - `multipeak` — a secondary maximum within 20% of the primary (aliasing)
   - `fail` — no significant peak (max C < 0.3) / insufficient overlap

   Deliberately *not* based on the correlation peak's width: smooth SN light
   curves always give a wide C(τ) peak, so width measures curve shape, not
   estimate precision — the GP-draw scatter σ(Δt) is the honest precision.
   Correlations are only evaluated at lags where the *bright* parts
   (> 30% of peak) of both curves overlap; normalized correlation of two
   flat zero-flux baselines is spuriously high and creates fake peaks at
   extreme lags otherwise.
7. **Flux ratio:** ratio of GP peak fluxes, per band, after shifting B by τ̂;
   report the per-band values and their inverse-variance-weighted mean. This
   is a *total* (macro × micro) magnification-ratio proxy.

### 2.5 Outputs

Per system (§6.3 for the schema): for each non-reference image, `dt_gp`,
`dt_gp_err`, `dt_gp_per_band` (dict), `flux_ratio`, `flux_ratio_per_band`,
`quality` flag; plus the reference image label and wall time.

Driver: `scripts/run_gp_population.py` [TO BUILD] — same joblib pattern as
the existing drivers, writes `gp_benchmark.ecsv` with fitted-vs-true columns
so Stage 1 gets the same residual benchmark treatment as Stage 3.

### 2.6 How it feeds the next stages

- **→ Stage 3 (the priming handoff, the first real payoff):** convert
  (`dt_gp`, `dt_gp_err`) into per-image t0 windows:
  `t0_img ∈ t0_ref + dt_gp ± max(4·dt_gp_err, 10 d)`. Concretely, in
  `measure_one()` this replaces the `sncosmo.fit_lc` self-recentering of
  `fast` mode and sets `t0_window` per system instead of the fixed 30 d in
  `FIT_PRESETS["fast"]`. For the BayeSN path it replaces the peak-offset
  initialization of `dt_*` bounds in `fit_system()` (currently
  `peak offset ± delay_window=40 d`).
  **Routing rule:** `quality == good` → fast mode with the GP window;
  `broad`/`multipeak`/`fail` → robust mode (wide bounds). The flag doubles as
  a "hard system" marker in all downstream analysis.
- **→ Stage 2:** `dt_gp`, `dt_gp_err`, `flux_ratio`, per-band delay scatter,
  and the quality flag (one-hot) become per-system scalar features (§3.3).
  The GP peak time of the *reference image* becomes the phase zero-point for
  tokenization (§3.2 — this exact choice is load-bearing; see the warning).

---

## 3. Stage 2 — Transformer (amortized inference)

### 3.1 Purpose

Millisecond-scale prediction of Δt, magnification ratios, microlensing
descriptors, and SN parameters — each with a calibrated uncertainty — from raw
photometry plus Stage-1 hints. Trained supervised on Stage-0 labels.

### 3.2 Input treatment — tokenization (`roman_td/tokenize.py` [TO BUILD])

One token per photometric measurement:

```
token = [ phase_norm, flux_norm, fluxerr_norm, band_onehot(6), image_onehot(4), is_real ]
```

Exact treatment, in order:

1. **Phase zero-point — one common reference per SYSTEM.**
   `phase = (mjd − t_peak,ref) / (1 + z_source)` where `t_peak,ref` is the GP
   peak of the *reference image only*.

   > **WARNING (delay-destroying bug if done wrong):** do NOT phase each
   > image relative to its own peak. That subtracts the delay out of the
   > input; the network would have nothing left to learn the delay from and
   > would parrot the GP hint. All images share the reference image's
   > zero-point, so a non-reference image's tokens sit ≈ Δt/(1+z) away from
   > the reference's — that offset IS the signal.

2. **Flux normalization — per system, internal reference.**
   `flux_norm = flux / max(flux over all images, bands, epochs of this system)`;
   `fluxerr_norm = fluxerr / (same denominator)`. Never normalize via a
   cosmological distance modulus — that divides out the magnification you are
   trying to measure. Absolute scale information re-enters only through
   `z_source` as a scalar feature.
3. **Band one-hot** over the 6 Roman filters (F062…F184), fixed ordering,
   zeros for bands absent from a survey tier.
4. **Image one-hot** over slots 1–4, assigned in the order of the truth
   record's `images` list (reference image always slot 1).
5. **Padding/masking:** pad every system to `L_max` tokens (choose the 99th
   percentile of token counts over the training set, ≈ 400–600 for 5-day
   cadence); `is_real = 0` on padding; attention-mask padding out.
6. **Non-detections are data:** epochs where the survey observed but flux/err
   < 1σ still become tokens (flux_norm near 0 with its real error). An early
   non-detection of image B pins the delay lower bound.

### 3.3 Per-system scalar features (fused after pooling)

`[ z_source, z_lens, n_images, dt_gp (per image slot), dt_gp_err,
flux_ratio, gp_band_scatter, gp_quality_onehot(4) ]`

**GP-hint dropout:** during training, with probability 0.2 zero all GP-derived
features (and set the quality one-hot to `fail`). This forces the network to
solve the problem from tokens alone and treats the hint as refinement, so a
wrong GP estimate at inference cannot fully steer the prediction.

**Redshift assumption:** both `z_lens` and `z_source` are available for every
system (survey design guarantees deflector and source redshifts), so they are
fed as exact scalar features with no missing-value handling and no jitter
augmentation. The rest-frame phase division by (1 + z_source) in tokenization
(§3.2) can likewise treat z_source as exact.

### 3.4 Architecture

- Token embedding: linear → d_model = 128.
- Encoder: 4–6 pre-norm transformer blocks, 8 heads, GELU, dropout 0.1.
- Pooling: one learned `[CLS]` summary token.
- Fusion: concat pooled vector with scalar features → 2-layer MLP → shared
  representation `h` (256-d).
- Size target ≈ 1–3 M parameters. Do not scale up until the gradient-boosted
  baseline (§5.3) and this size are both saturated.

### 3.5 Output heads — fixed-size parameterization

> Variable image count is handled by predicting **per-image quantities
> relative to the reference image**, with a per-slot validity mask — never
> "per pair" (variable-size output). This mirrors SNTD's series
> parameterization (`dt_img`, `mu_img/mu_ref`), making the Stage-3 handoff a
> direct substitution.

| Head | Output (per non-reference image slot i ∈ {2,3,4}) | Activation / range |
|---|---|---|
| Delay | `dt_i`, `log σ_dt,i` | linear; days, observer frame |
| Magnification ratio | `log(μ_i/μ_ref)`, `log σ` | linear in log-ratio |
| Microlensing (per image incl. ref) | amplitude `A_micro,i` (mag), chromatic slope `dA/dband_i`, `log σ` each | linear; 0 when absent |
| SN parameters (per system) | `theta_hat`, `A_V_hat`, `log σ` each | linear |

All heads read the shared `h` (joint prediction lets microlensing/color/dust
constrain each other). Masked slots (absent images) contribute zero loss.

### 3.6 Training

- **Loss:** sum over heads of Gaussian negative log-likelihood
  `0.5·[(y−ŷ)²/σ² + log σ²]`, with per-target masks (absent image slots;
  micro targets are 0 — not masked — in tiers 0–1, so the network learns to
  *report* zero, and masked only where truly undefined).
- **Curriculum:** epochs 1–N on tiers 0–1 only, then phase in tier 2, then 3.
- **Split discipline:** train/val/test split **by lens system**, never by
  noise realization — multiple realizations of one system must land in the
  same split or the test set leaks.
- **Calibration (load-bearing, not optional):** on the validation split,
  compute the empirical coverage of the σ's (68%/95%); apply per-head
  temperature scaling (or conformal offsets) as a fixed post-processing step.
  If the delay residuals show multimodality (aliasing), upgrade the delay
  head to a 3-component mixture density — decide from the validation
  residuals, not in advance.
- **Model-swap robustness test:** train on BayeSN-simulated systems, evaluate
  on SALT-simulated systems (and vice versa). The degradation measures
  sim-dependence directly — report it alongside every accuracy number.

### 3.7 Outputs

Per system (§6.4): per-image-slot `dt`, `dt_err`, `mu_ratio`, `mu_ratio_err`,
micro amplitude/slope ± σ, SN params ± σ — all *post-calibration* — plus the
model version hash and the tier (training bookkeeping only).

### 3.8 How it feeds Stage 3

Same mechanical handoff as the GP (§2.6) but tighter:
`dt_i ± 4σ` → per-image t0 bounds; `theta_hat ± σ` → Gaussian prior on theta
in `fit_system()` (replacing the N(0,1) default); `A_V_hat` → `hostebv`
bound narrowing. Microlensing outputs do **not** feed Stage 3 (no microlensing
term in the fit model yet) — they are flagging-grade side products validated
against Stage-3 residuals.

At inference on real data the chain is: Stage 1 → Stage 2 → Stage 3 for every
system; Stage 2's numbers are the fast catalog, Stage 3's the publication
numbers on the subset where it is run.

---

## 4. Stage 3 — Physical fit (SNTD; the anchor)

### 4.1 Purpose

Simulation-independent, publication-grade delays and magnification ratios via
nested sampling of a physical SED model. Everything upstream exists to make
this stage start in the right place and therefore run fast.

### 4.2 Path A — SALT2-extended, `sntd_wrapper.measure_one()` (EXISTS)

**Inputs:** lens object, bands, z_source, cadence, depths, RNG, plus the fit
knobs (`t0_window`, `t0_pad_days`, `npoints`, `maxcall`) bundled as
`FIT_PRESETS["fast"]` = (30 d, 100, 8000) or `["robust"]` = (None, None, None).

**Treatment:**
1. Simulate photometry internally via `extract_light_curves()` (Stage-0 Path A).
2. Cuts: skip if max true delay > `max_delay_days` (150) or any image peak
   SNR < `min_image_snr` (10).
3. SNTD `parallel` method: fit SALT2-extended to each image independently
   with nested sampling; delay = difference of per-image fitted t0.
   - *robust*: t0 bounds span each image's significant-SNR detection window
     ± `t0_pad_days`; uncapped sampling.
   - *fast*: `sncosmo.fit_lc` pre-fit re-centers a `t0_window`-wide window
     per image; `npoints`/`maxcall` capped. **With Stage 1 built, the pre-fit
     re-centering is replaced by the GP window (§2.6) — this is the single
     highest-value integration in the whole architecture.**

**Outputs:** result dict → benchmark rows in `delay_benchmark_{mode}.ecsv`
(§6.5). **[TO BUILD, few lines]:** also extract the per-image fitted `x0` and
write `mu_ratio_fit = x0_i / x0_ref` — the magnification ratio is currently
computed by the sampler and thrown away (design doc §7).

### 4.3 Path B — BayeSN two-stage, `bayesn_wrapper.fit_system()` (EXISTS)

**Inputs:** MISN object + combined table (Stage-0 Path B), z, images,
`npoints_series`, `npoints_color` (500 default; 100–200 acceptable),
`delay_window` (40 d), `ncpu_fit`.

**Treatment:**
1. Reference image = brightest. `dt_img` bounds = peak-flux offset ± 40 d
   (→ replaced by GP/transformer windows when available).
2. **Series fit** (`nest_series_lc`): joint sampling of
   `t0, theta, hostebv, amplitude, dt_*` with `theta ~ N(0,1)` prior. Delays
   are *sampled parameters*, not derived.
3. **Prior tightening:** series posterior → Gaussian theta prior
   (med ± 5σ bounds) and 2σ-quantile t0 bounds for stage two — the in-repo
   template for how *all* stage-to-stage handoffs in this architecture work.
4. **Color fit** (`nest_color_lc`): adjacent-band color curves; `hostr_v`
   freed (color constrains it independently of amplitude); delays re-sampled.
   Falls back to series results if < 2 bands.

**Outputs:** per-image delay posteriors (median, ±1σ from quantiles), timing
per stage, ref label → `delay_benchmark.ecsv` + optional per-lens JSON
payloads.

### 4.4 Routing policy (which path, which mode)

```
GP quality == good      → SALT fast (GP-primed)          ~seconds–minute
GP quality == broad     → SALT robust                     ~minutes
GP quality == multipeak → SALT robust; flag for BayeSN
high-value systems      → BayeSN two-stage (GP-primed)    ~minutes–tens of min
(e.g. best H0 leverage, microlensing candidates from Stage 2)
```

The residual distribution *between* fast and robust on the same systems is
itself a deliverable (accuracy cost of speed), via
`scripts/compare_fit_modes.py` (EXISTS).

---

## 5. Validation & benchmarking (cross-cutting)

### 5.1 Metrics — computed identically for Stages 1, 2, 3

Per delay measurement, against the truth record:

- `residual = dt_fit − dt_true` (days) — already in the benchmark tables.
- **`frac_residual = residual / dt_true`** [TO ADD] — the H0-relevant number
  (target ~1–2%); a 1-day error means opposite things at Δt = 5 d vs 60 d.
- Coverage: fraction of systems with |residual| < 1σ_reported (target 0.68).
- Binned by: `dt_true / cadence` (delays < 2 cadence steps are the failure
  regime), z_source, n_images, tier, GP quality flag.
- Wall time per system (already recorded by both drivers).

### 5.2 The three load-bearing tests

1. **Calibration** (per stage that reports σ): coverage plots before/after
   temperature scaling. An overconfident σ is worse than none.
2. **Model-swap:** BayeSN-sim ↔ SALT-sim train/test cross (§3.6).
3. **Matched-pair ablation:** tier-2 vs tier-1 twins (same lens, same SN,
   same noise seed, microlensing on/off) → how much microlensing degrades Δt
   and whether the micro head recovers the injected amplitude. This is the
   referee-facing controlled experiment; the physical fits cannot produce it.

### 5.3 The boring baseline (build before the transformer)

Gradient-boosted trees on Stage-1 summary features (dt_gp, err, per-band
scatter, flux ratios, n_obs, peak SNRs, z's, quality flag) predicting the same
targets. If it meets the Δt accuracy bar, the transformer is optional for
delays; either way it sets the number the transformer must beat.

**Status: BUILT** — `scripts/gbt_baseline.py`; method and results in
`documents/gbt_baseline_explained.md`.

---

## 6. Data contracts (exact schemas)

### 6.1 Canonical photometry table (Stage 0 → 1, 2, 3)

Astropy Table (ECSV on disk), one row per (image, epoch, band):

| column | dtype | definition |
|---|---|---|
| `mjd` | f8 | observer-frame time |
| `filter` | str | sncosmo band name (e.g. `f129`) |
| `flux` | f8 | at `zp` = 25.0, AB |
| `fluxerr` | f8 | 1σ, from 5σ depth / 5 |
| `zp` | f8 | 25.0 always |
| `zpsys` | str | `"ab"` |
| `image` | str | `image_1` … `image_4`; ordering = truth record `images` list |

This is Path B's existing format. Path A emits per-image tables with columns
`time`/`band`; **[TO BUILD]** a ~10-line adapter `to_canonical(image_tables)`
in `roman_td/simulate.py` rather than touching `extract_light_curves`.

### 6.2 Truth record (Stage 0 → training + all benchmarks)

JSON per system, `lens_{i:05d}__truth.json`:

```json
{
  "lens_index": 42, "tier": 2,
  "z_lens": 0.41, "z_source": 1.12,
  "images": ["image_1", "image_2"],
  "delays": [0.0, 23.7],
  "mu_macro": [4.1, 2.3],
  "micro": {"image_1": {"mjd": [...], "F129": [...], ...}, ...},
  "sn_params": {"theta": 0.31, "hostebv": 0.05, "hostr_v": 3.1, "amplitude": 1.2e-4},
  "sim_config": {"survey": "time_domain_deep", "cadence": 5.0,
                  "depths": {...}, "seed": 42, "sim_model": "bayesn"}
}
```

Delays and μ are **post-detection-cut, re-referenced** values (§1.3 Path B
step 5). `micro` curves sampled on the observation grid; all-zeros for
tiers 0–1.

### 6.3 Stage 1 output (GP → 2, 3)

JSON/dict per system:

```json
{
  "lens_index": 42, "ref_image": "image_1", "t_peak_ref": 60012.3,
  "per_image": {
    "image_2": {"dt_gp": 22.1, "dt_gp_err": 3.4,
                 "dt_per_band": {"f106": 21.0, "f129": 23.5},
                 "flux_ratio": 0.55, "flux_ratio_err": 0.06,
                 "quality": "good"}
  },
  "wall_time_s": 4.2
}
```

### 6.4 Stage 2 output (transformer → 3 + catalog)

Per system, per image slot: `dt`, `dt_err`, `log_mu_ratio`, `log_mu_ratio_err`,
`micro_amp`, `micro_slope` (+ errs), `theta_hat`, `av_hat` (+ errs),
`model_version`, `calibration_version`. All σ post-calibration.

### 6.5 Stage 3 output (benchmark tables — EXIST)

`delay_benchmark_{fast,robust}.ecsv` / `delay_benchmark.ecsv`, one row per
non-reference image: `lens_index, image, z_lens, z_source, n_images,
true_delay, fit_delay, fit_err_lo, fit_err_hi, residual, fit_mode,
fit_time_s`. **[TO ADD]:** `true_mu_ratio, fit_mu_ratio, frac_residual,
gp_quality` columns as the upstream stages land.

---

## 7. Build order with dependencies

| # | Task | Depends on | Est. size | Payoff |
|---|---|---|---|---|
| 1 | `x0`-ratio extraction in `measure_one` + `mu_ratio` benchmark columns | — | few lines | magnification benchmark unblocked |
| 2 | `frac_residual` + binned metrics in benchmark analysis | — | small | H0-relevant accuracy visible |
| 3 | Canonical-table adapter (§6.1) | — | ~10 lines | one schema everywhere |
| 4 | **Stage 1 GP** (`crosscorr.py`) + `run_gp_population.py` | 3 | days | model-free Δt in seconds |
| 5 | GP-primed fast mode in `measure_one` + routing rule | 4 | small | fast mode trustworthy; big wall-time win |
| 6 | Truth records + tier-0/1 training set builder | 3 | days | training data flows |
| 7 | Microlensing injection (tier 2) | 6 | the hard one | the novel labels |
| 8 | GBT baseline on GP features | 4, 6 | days | the number to beat |
| 9 | Tokenizer + transformer + calibration | 6, 7, 8 | weeks | amortized inference |
| 10 | Model-swap + matched-pair validation | 9 | days | the referee-facing evidence |

Items 1–5 involve no ML and already improve the current pipelines; items 6–10
are the ML track and can proceed in parallel once 3 lands.
