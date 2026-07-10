# Lensed SN Ia Time-Delay Pipeline — Architecture Specification

Companion to `lensed_sn_pipeline_design.md` (the *why*). This document is the
*how*: what each stage takes in, what it does, what it writes, and how the
pieces connect. **Last updated 2026-07-09.** Status markers: **BUILT** (code
exists, has run, results on disk) / **UNBUILT** (specified only).

Deep dives live in their own docs and are not repeated here:
`transformer_explained.md` (design), `transformer_training_explained.md`
(how it learns), `gbt_baseline_explained.md`, `fixes.md` (bug ledger +
pending fixes).

## Repository map (all files exist)

```
time_delays/
├── data/                            # slsim lens population pickles (input of everything)
├── roman_td/                        # library
│   ├── simulate.py                  # Stage 0: BayeSN sim + HLTDS survey spec
│   ├── crosscorr.py                 # Stage 1: GP cross-correlation
│   ├── tokenize.py                  # Stage 2: tokens + scalar features (+ DT_SCALE)
│   ├── ml_data.py                   # Stage 2: PyTorch Dataset (+ crop augmentation)
│   ├── transformer.py               # Stage 2: model + loss
│   ├── sntd_wrapper.py              # Stage 3: SALT path (+ Path-A sim)
│   ├── bayesn_wrapper.py            # Stage 3: BayeSN two-stage (INFEASIBLE — kept for record)
│   ├── bayesncosmo.py               # BayeSN as an sncosmo source
│   └── paths.py                     # repo-relative output paths
├── scripts/
│   ├── run_sntd_population.py       # benchmark driver: SALT fast/robust
│   ├── run_gp_population.py         # benchmark driver: GP alone / GP-primed SALT
│   ├── run_bayesn_population.py     # benchmark driver: BayeSN (feasibility record)
│   ├── compare_fit_modes.py         # four-way comparison plots/table
│   ├── build_training_set.py        # Stage 0+1 → training examples on disk
│   ├── train_transformer.py         # Stage 2 training driver
│   └── gbt_baseline.py              # boring baseline (GBT on GP features)
└── outputs/
    ├── benchmarks/*.ecsv            # one row per (system, image): fitted vs true
    ├── training/tier1_{deep,wide}/  # training sets (spec-consistent, post-fixes)
    └── models/<run_name>/           # best.pt + config.json + history.csv per run
```

Environments: simulation/fitting/building run in `sntd_bayesn`
(`/home/epadill/miniconda3/envs/sntd_bayesn/bin/python`); ML training runs in
`roman_ml` (torch). The two never mix: `build_training_set.py` needs the
fitting stack, `train_transformer.py` only torch+numpy+astropy.

---

## 0. How the pieces connect — the three chains

The four stages (0 sim, 1 GP, 2 transformer, 3 physical fit) are wired into
three concrete chains. Two run today; the third is the assembled product.

### Chain A — benchmark chain (BUILT, results on disk)

*Question it answers: how accurate/fast is each classical method?*

```
data/*.pkl ─▶ run_sntd_population.py ──▶ outputs/benchmarks/delay_benchmark_{fast,robust}.ecsv
data/*.pkl ─▶ run_gp_population.py  ──▶ outputs/benchmarks/gp_only_benchmark.ecsv
        (same script, --fit flag) ────▶ outputs/benchmarks/delay_benchmark_gp.ecsv  (GP-primed SALT)
                                                  │
                          compare_fit_modes.py ◀──┘   (four-way table + plots)
```

Measured (185 common systems, pre-spec cadence — internally consistent,
optimistic vs HLTDS spec):

| method | delay P68 | wall time / system |
|---|---|---|
| GP alone | 2.01 d (good subset 1.65 d) | 4 s |
| GP-primed SALT fast | 1.48 d (good subset 1.37 d) | ~270 s |
| SALT fast (unprimed) | 1.88 d | ~260 s |
| SALT robust | 1.14 d | ~3300 s |

Provisional GP bar on spec-realistic sims (42-example check): P68 2.17 d all,
**1.43 d on `good`** — the number Stage 2 must beat; to be re-measured on the
full deep build.

### Chain B — ML training chain (BUILT, runs in progress)

*Question it answers: can a network match/beat the above in milliseconds?*

```
data/roman_deep*.pkl ─▶ build_training_set.py          (env: sntd_bayesn)
                          │  per lens: Stage-0 sim (§1) → detection cuts
                          │            Stage-1 GP (§2) on each noise realization
                          ▼
        outputs/training/tier1_deep/tier1/
            index.ecsv                      (one row per example: split, gp_ok, paths)
            lens_XXXXX_rY__phot.ecsv        (photometry, canonical schema §6.1)
            lens_XXXXX_rY__truth.json       (truth + GP result, §6.2–6.3)
                          │
                          ▼
        train_transformer.py --name <run>   (env: roman_ml)
            reads examples via roman_td/ml_data.py + tokenize.py
            (same tokenizer code that inference will use — no train/serve skew)
                          ▼
        outputs/models/<run>/best.pt + history.csv + config.json
```

Data on disk: `tier1_deep` = 8,286 examples / 2,762 systems (3 noise
realizations each; splits by lens system 6657/879/750), `tier1_wide` =
5,055 / 1,685. (`outputs/training/tier1` is the pre-fix set — invalid, see
fixes.md t0 bug.) Runs so far: `deep_full_v1`/`deep_crop_v1` (07-08, delay
head stalled — units bug, fixed by `DT_SCALE`), `deep_full_v2`/`deep_crop_v2`
(07-09, in progress, stall resolved). Pending one-at-a-time fixes (b)(c)(d):
see fixes.md "Pending".

### Chain C — inference chain (the product; wiring UNBUILT)

*What runs on real (or held-out) data once assembled:*

```
photometry ─▶ Stage 1 GP ─▶ Stage 2 transformer ─▶ routing (§4.4) ─▶ Stage 3 SNTD
                 hints         fast catalog:            quality-based      publication-grade
                 + windows     Δt, μ, micro, ±σ         fast/robust        Δt posteriors
```

Every stage exists; what's missing is the driver that chains them and the
post-training calibration pass (§5.2). Stage 2's output is the catalog for
*all* systems; Stage 3 runs on the subset worth CPU-hours, started inside
Stage 2's tight windows.

---

## 1. Stage 0 — Simulation (BUILT: `roman_td/simulate.py`)

**Purpose:** per lens system, (a) a realistic multi-image multi-band Roman
photometry table, (b) truth labels (delays, magnifications, micro curves, SN
parameters). (a) feeds Stages 1–3; (a)+(b) feed training and every benchmark.

**Inputs:** slsim lens pickle (provides redshifts, `point_source_arrival_times()`,
`point_source_magnification()`); survey tier; RNG seed
(`default_rng(seed + lens_index)` — reproducible, worker-independent).

**Two simulators:**
- **Path A (SALT):** `sntd_wrapper.extract_light_curves()` — lenstronomy
  ray-traced lensed magnitudes; used by the Chain-A benchmarks.
- **Path B (BayeSN):** `simulate.simulate_photometry()` — draw one BayeSN SN
  (`theta~N(0,1)`, `hostebv~Exp(0.1)`, amplitude set for luminosity distance),
  then per image k: `mu_k · bandflux(band, t − dt_k)` + depth-map noise.
  Used by Chain B (the training set).

**Detection cuts (Path B):** image kept iff peak SNR ≥ 5 and ≥ 5 epochs;
system kept iff ≥ 2 images survive. Kept delays re-zeroed to the first kept
image — downstream must always use the truth record's `images` list, never
assume `image_1`. Yield vs spec depths: ~27% of deep, ~17% of wide systems
detectable — physics (demagnified counter-images at high z), not cut tuning.

**HLTDS Core Community Survey spec (encoded 2026-07-07):** per-tier anchor
filter every 5 d + four filters every 10 d; per-visit 5σ depths from CCS
exposure times × WFI 1-hr sensitivities, m5(t) = m5(1hr) − 1.25·log₁₀(3600/t).
In `simulate.SURVEY_CADENCE` / `SURVEY_EXPTIME` / `SURVEY_DEPTH_5SIG`;
`build_training_set.py` uses these by default (`--cadence` = flat override
for controlled experiments). Legacy `DEFAULT_DEPTH_5SIG` dicts remain in
`simulate.py`/`sntd_wrapper.py` for the old benchmarks only.

| Tier | 5-d anchor | 10-d filters | depths (anchor first) |
|---|---|---|---|
| Wide | F062 | F087 F106 F129 F158 | 25.75 / 25.60 25.63 25.88 26.16 |
| Deep | F087 | F106 F129 F158 F184 | 26.04 / 26.24 26.26 26.35 26.52 |

**Everything simulated before 2026-07-07 is NOT spec-consistent** (flat 5-d
cadence, 0.6–1.5 mag too deep; Path B additionally had the t0 double-count
bug — see fixes.md). Chain-A benchmarks are internally valid but optimistic.

**Path C — microlensing injection (UNBUILT, tier 2):** per image, draw a
magnification map (κ, γ, s from the macro model), convolve with the growing
chromatic SN photosphere → `μ_micro(t, band)`; multiply into the flux before
noise; store the curve as a truth label. Tier flag per system: 0 = SN only,
1 = +macro (current sets), 2 = +micro, 3 = +milli. One network trains on all
tiers; the flag is for curriculum/ablation, never a network input.

---

## 2. Stage 1 — GP cross-correlation (BUILT: `roman_td/crosscorr.py`)

**Purpose:** model-free coarse Δt in ~4 s/system: GP-interpolate each image's
light curve, cross-correlate pairs. Not built for accuracy — built to place
Stage 3's search window and give Stage 2 hint features. Measured anyway at
P68 = 2.01 d (§0 Chain A).

**Algorithm** (per pair, reference = brightest image, per shared band):
Matérn-3/2 GP per image (length scale bounded [3, 40] d) → evaluate on a
0.5-d grid → inverse-variance-weighted correlation vs lag, **scored only
where the bright parts (> 30% of peak) of both curves overlap** (flat
baselines correlate spuriously) → parabolic peak refinement → uncertainty
from 50 Cholesky posterior draws × bands (per-band scatter kept — chromatic
disagreement is a microlensing indicator) → flux ratio from shifted GP peaks.

**Quality flag** from draw scatter + rival peaks (never peak width — SN
curves always correlate broadly): `good` (σ ≤ 5 d, single peak), `broad`,
`multipeak` (secondary ≥ 80% of primary), `fail`.

**Feeds:**
- **→ Stage 3:** t0 window per image = `t0_ref + dt_gp ± max(4σ, 10 d)`,
  replacing fast mode's self-recentering. This is the measured 1.88 → 1.48 d
  improvement in Chain A — priming beats unprimed fast in both accuracy and
  it's the routing anchor (§4.4).
- **→ Stage 2:** `dt_gp`, `dt_gp_err` (as days/`DT_SCALE`, §3.3), flux
  ratios, band scatter, quality one-hot become scalar features; the
  *reference image's* GP peak time is the phase zero-point of tokenization.

---

## 3. Stage 2 — Transformer (BUILT: `tokenize.py`, `ml_data.py`, `transformer.py`)

**Purpose:** millisecond amortized inference — Δt, magnification ratios,
microlensing descriptors, SN parameters, each with a learned σ — from raw
photometry + Stage-1 hints. Full design rationale in `transformer_explained.md`;
optimizer/loss mechanics in `transformer_training_explained.md`.

### 3.1 Tokenization (one token per photometric point)

```
token = [ phase, flux_norm, fluxerr_norm, band_onehot(6), image_onehot(4), is_real ]
```

- **Phase zero-point — one per SYSTEM:** `phase = (mjd − t_peak,ref)/(1 + z_source)`,
  `t_peak,ref` = GP peak of the reference image only.
  > **WARNING (delay-destroying if done wrong):** never phase each image to
  > its own peak — that subtracts the delay out of the input. All images
  > share the reference zero-point; the inter-image token offset IS the
  > delay signal.
- **Flux:** normalized by the system's brightest point (preserves
  magnification ratios; never normalize by distance modulus).
- Padding to `l_max = 512` with attention mask; non-detections are real
  tokens (an early non-detection of image B pins the delay lower bound).

### 3.2 Scalar features (17, fused after pooling)

`[z_source, z_lens, n_images, dt_gp×3, dt_gp_err×3, flux_ratio×3,
gp_band_scatter, quality_onehot(4)]`. Delay-valued entries are stored as
**days / `DT_SCALE` (= 100)** — the same unit the delay head predicts in.
Raw-day targets stalled the v1 runs (heteroscedastic-NLL σ-inflation; see
fixes.md 07-09). Known wart: hint slots are currently misaligned one slot vs
targets — pending fix (b).

### 3.3 Training-time augmentations (train split only)

- **Noise realizations** (×3 per lens, same SN, different photometric noise) —
  baked into the training set by the builder.
- **GP-hint dropout** (p = 0.2): zero the GP features, quality → `fail`; the
  hint must stay a refinement, not a crutch.
- **Observer-window crop** (`--crop_prob`, off by default; 0.7 in crop runs):
  clip each example to ONE shared random `--crop_window` (365 d) MJD window —
  shared because Roman sees all images every visit, so trailing-image
  truncation emerges from the delays automatically. Guards: ref image keeps
  ≥ 5 pts, ≥ 2 images survive, else no crop; an image left < 3 pts gets its
  dt/logmu loss masked; a crop removing > 20% of points also drops the GP
  hint (full-curve GP would leak unseen data). Val/test always full curves.

Both z's are guaranteed known for every system (survey design) — exact
features, no missing-value handling.

### 3.4 Architecture and loss (0.91 M parameters)

Linear embed (14 → 128) → learned CLS token → 4 pre-norm encoder blocks
(8 heads, FF 512, dropout 0.1) → CLS ⊕ scalars → fusion MLP (145 → 256 → 256)
→ four heads, each emitting (value, log σ): delays (3 non-ref slots),
log μ-ratios (3), micro amp+slope (4 images), SN params (theta, host E(B−V)).
Loss = masked Gaussian NLL summed over heads, equal weights; micro targets
in tiers 0–1 are zeros and *not* masked — "no microlensing" must be reported.
Do not scale the model up until this size and the GBT baseline are saturated.

**Delay unit contract:** the network reads and writes delays in
days/`DT_SCALE`; `train_transformer.py` (and any future inference driver)
multiplies outputs and σ by `DT_SCALE` before reporting days.

### 3.5 Training runs so far

| run | date | config | outcome |
|---|---|---|---|
| tier1_v1, smoke | 07-06 | 384-example set | invalid (t0 sim bug) |
| deep_full_v1 / deep_crop_v1 | 07-08 | 60 ep, batch 64 / +crop 0.7 | delay head stalled: val P68 ≈ 40 d (raw-day targets; fixes.md 07-09) |
| deep_full_v2 / deep_crop_v2 | 07-09 | same + DT_SCALE fix | in progress; delay NLL descending from epoch 0 |

Remaining known fixes, applied ONE per retrain (v3 = +b, v4 = +c):
(b) hint-slot alignment, (c) checkpoint on val_dt not val_total,
(d) optional σ warmup — fixes.md "Pending".

### 3.6 Post-training (both UNBUILT, load-bearing)

- **Calibration:** temperature-scale each head's σ on validation to hit 68%
  coverage; upgrade the delay head to a small mixture density *only if* the
  validation residuals show aliasing multimodality.
- **Evaluation battery:** hints-off validation (is it more than a GP echo?),
  truncation-stratified scoring (full / post-peak / trailing-clipped),
  model-swap (train BayeSN-sim ↔ test SALT-sim).

### 3.7 Feeds Stage 3

Same handoff as the GP but tighter: `dt ± 4σ` → per-image t0 bounds;
`theta ± σ` → Gaussian prior; E(B−V) → hostebv bound narrowing. Micro outputs
do NOT feed Stage 3 (no micro term in the fit model) — they are
flagging-grade side products validated against Stage-3 residuals.

---

## 4. Stage 3 — Physical fit (SNTD; the anchor)

**Purpose:** simulation-independent, publication-grade delays via nested
sampling of a physical SED model. Everything upstream exists to make this
stage start in the right place and therefore run fast.

### 4.1 Path A — SALT2-extended (BUILT: `sntd_wrapper.measure_one()`)

SNTD `parallel` method: each image fit independently; delay = difference of
fitted t0. Modes: `robust` (wide bounds, uncapped; P68 1.14 d @ 55 min) and
`fast` (windowed, capped; 1.88 d unprimed → **1.48 d GP-primed** @ ~4.5 min).
Cuts: max true delay 150 d, min image SNR 10. [TO ADD, few lines]: extract
per-image fitted `x0` → `mu_ratio_fit` benchmark column (currently computed
by the sampler and discarded).

### 4.2 Path B — BayeSN two-stage (BUILT, verdict: **INFEASIBLE** 2026-07-06)

`bayesn_wrapper.fit_system()`: joint series fit (delays as sampled
parameters) → prior tightening → color fit. With every mitigation (GP-primed
windows, log-uniform amplitude, 200k-call cap): 5.4 h/system, caps hit,
bound-pinned delays with zero-width errors. Root cause: SNTD's 0.1-d phase
rounding creates likelihood plateaus that kill joint nested sampling at these
SNRs (fixes.md). Code kept as record. **Forward path (UNBUILT):** BayeSN SED
through SNTD's *parallel* method — per-image fits sidestep the joint
degeneracy.

### 4.3 Routing policy (UNBUILT as code; the rule)

```
GP/transformer quality good       → SALT fast, primed windows    (~minutes)
quality broad or multipeak        → SALT robust                  (~1 h, flagged "hard")
high-value systems (H0 leverage,  → BayeSN-parallel, primed      [UNBUILT]
  microlensing candidates)
```

The fast-vs-robust residual distribution on shared systems is itself a
deliverable (accuracy cost of speed): `compare_fit_modes.py`.

---

## 5. Validation & benchmarking (cross-cutting)

### 5.1 Metrics — identical for Stages 1, 2, 3

`residual = dt_fit − dt_true` (d); P68 of |residual|; % < 2 d; coverage
(|res| < 1σ, target 68%); wall time. [TO ADD]: `frac_residual = residual /
dt_true` — the H0-relevant number (target 1–2%); binning by dt_true/cadence,
z_source, n_images, tier, GP quality.

### 5.2 The three load-bearing tests

1. **Calibration** before/after temperature scaling — an overconfident σ is
   worse than none.
2. **Model-swap** BayeSN-sim ↔ SALT-sim — measures sim-dependence; report
   next to every accuracy number.
3. **Matched-pair ablation** (tier-2 vs tier-1 twins: same lens, SN, noise
   seed; micro on/off) — how much microlensing degrades Δt and whether the
   micro head recovers the injected amplitude. The referee-facing controlled
   experiment; physical fits cannot produce it.

### 5.3 The boring baseline (BUILT: `scripts/gbt_baseline.py`)

Gradient-boosted trees on Stage-1 summary features predicting the *residual*
(true − dt_gp; predicting the absolute delay wastes capacity re-learning the
identity — fixes.md 07-06). Sets the number the transformer must beat;
method and results in `gbt_baseline_explained.md`.

---

## 6. Data contracts (exact schemas)

### 6.1 Canonical photometry table (Stage 0 → 1, 2, 3)

ECSV, one row per (image, epoch, band): `mjd` (f8, observer frame), `filter`
(sncosmo name, e.g. `f129`), `flux`/`fluxerr` (ZP 25 AB; σ = depth_5σ flux/5),
`zp` (25.0), `zpsys` ("ab"), `image` (`image_1`…`image_4`, ordering = truth
record's `images`). This is Path B's native format; Path A emits per-image
tables (`time`/`band`) — adapt, don't modify.

### 6.2 Truth record (Stage 0 → training + benchmarks), one JSON per example

```json
{ "lens_index": 42, "tier": 1, "z_lens": 0.41, "z_source": 1.12,
  "images": ["image_1","image_2"], "delays": [0.0, 23.7], "mu_macro": [4.1, 2.3],
  "micro_amp": {"image_1": 0.0, "image_2": 0.0},
  "sn_params": {"theta": 0.31, "hostebv": 0.05, "hostr_v": 3.1, "amplitude": 1.2e-4},
  "sim_config": {"survey": "time_domain_deep", "cadence": {"F087": 5.0, "...": 10.0},
                 "depths": {"F087": 26.04, "...": 0}, "seed": 42},
  "gp": { ...Stage-1 result, §6.3... } }
```

Delays/μ are **post-detection-cut, re-referenced** values. The builder embeds
the Stage-1 GP result under `"gp"` so training needs no second pass.

### 6.3 Stage 1 output (GP → 2, 3)

Per system: `ref_image`, `t_peak_ref`, `wall_time_s`, and per non-reference
image `{dt_gp, dt_gp_err, dt_per_band, flux_ratio, flux_ratio_err, quality}`.
Stored in physical days here; conversion to days/`DT_SCALE` happens only
inside `scalar_features()`.

### 6.4 Stage 2 output (transformer → 3 + catalog)

Per system, per image slot: `dt`, `dt_err`, `log_mu_ratio` (+err),
`micro_amp`, `micro_slope` (+errs), `theta_hat`, `av_hat` (+errs), model +
calibration version. Reported in **days** (heads × `DT_SCALE`), σ
post-calibration.

### 6.5 Stage 3 / benchmark tables (`outputs/benchmarks/*.ecsv`)

One row per non-reference image: `lens_index, image, z_lens, z_source,
n_images, true_delay, fit_delay, fit_err_lo, fit_err_hi, residual, fit_mode,
fit_time_s` (+ `gp_time_s`, quality in the GP tables). [TO ADD]:
`true_mu_ratio, fit_mu_ratio, frac_residual` as §4.1/§5.1 land.

---

## 7. Status ledger

**Done** (dates = when verified working):

| piece | date |
|---|---|
| Stage 3 SALT fast/robust + benchmarks | pre-07 |
| Stage 1 GP + driver + benchmark | 07-05 |
| GP-primed fast mode (the 1.88 → 1.48 d win) | 07-06 |
| GBT baseline (residual form) | 07-06 |
| BayeSN two-stage feasibility → INFEASIBLE verdict | 07-06 |
| HLTDS spec cadences/depths in sim + builder | 07-07 |
| t0 double-count sim bug found + fixed | 07-07 |
| Training-set builder; tier1_deep + tier1_wide built | 07-07/08 |
| Tokenizer, Dataset (+crop aug), transformer, training driver | 07-07/08 |
| v1 stall diagnosed → DT_SCALE fix (a); v2 runs launched | 07-09 |

**Remaining, in priority order:**

| # | task | blocked by |
|---|---|---|
| 1 | v2 verdict vs GP bar; fixes (b) hint alignment, (c) val_dt checkpointing, (d) σ warmup — one per retrain | v2 runs finishing |
| 2 | Evaluation battery: hints-off val, truncation-stratified, GP-bar re-measure on full spec build | 1 |
| 3 | Calibration pass (temperature scaling; mixture head only if aliasing seen) | 1 |
| 4 | Deep+wide mixed training (~60/40 realistic ratio) + per-tier metrics | 1 |
| 5 | Dynamic batch padding (2–4× training speedup) | — |
| 6 | Microlensing injection → tier-2 set; matched-pair ablation | — (the hard one) |
| 7 | BayeSN-parallel fitting path (GP/transformer-primed) | — |
| 8 | Inference driver wiring Chain C + routing rule as code | 1–3 |
| 9 | `mu_ratio` extraction + `frac_residual` benchmark columns | — (few lines) |
| 10 | Model-swap robustness test | 6 |
