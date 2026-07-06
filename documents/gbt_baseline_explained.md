# The GBT Baseline, Explained

What `scripts/gbt_baseline.py` does and why it exists. Companion to
`gp_crosscorr_explained.md` (Stage 1, which feeds it) and
`transformer_explained.md` (Stage 2, which must beat it). This is the
"boring baseline" of the architecture doc §5.3 — the first actual machine
learning in the pipeline, deliberately built *before* the transformer.

---

## 1. One-paragraph summary

A gradient-boosted-tree (GBT) model learns to predict the **true** time
delay from the GP cross-correlator's summary outputs plus basic observables
(~15 numbers per system). It is a learned correction-and-calibration layer
on top of the GP: it can fix systematic biases and recalibrate errors, but
it never sees the raw light curves. Its accuracy therefore measures how
much delay information the *summary statistics* contain — and the gap
between it and the future transformer (which sees all ~400 raw photometric
points) measures the value of the raw data. Cost: one script, minutes of
training, no new data.

## 2. What a gradient-boosted tree is

- A **decision tree** is a stack of if/else questions on the features
  ("`dt_gp_err` > 4.2? → `z_source` > 2.9? → predict X"). One tree is crude.
- **Gradient boosting** builds an ensemble *sequentially*: fit tree 1, fit
  tree 2 to tree 1's errors, tree 3 to the remaining errors, … 500 rounds.
  Each tree is deliberately weak (few branches; a 0.05 learning rate
  shrinks its vote), which prevents memorizing noise.
- Why this family for a baseline: it is the standard workhorse for
  *tabular* data (a fixed set of named numbers per example), needs no
  feature scaling, handles missing values natively (`dt_gp_err` and
  `band_scatter` contain NaNs), trains in minutes, and is hard to
  misconfigure. Implementation: sklearn `HistGradientBoostingRegressor`.

## 3. The exact learning problem

| | |
|---|---|
| Training data | `scripts/gp_only_benchmark.ecsv` — 1062 delays, 1000 lens systems |
| Features (15) | `dt_gp`, `dt_gp_err`, `band_scatter`, `n_bands`, `flux_ratio`, `flux_ratio_err`, `snr_ref`, `snr_img`, `z_lens`, `z_source`, `n_images`, quality flag (4 one-hots) |
| Target | `true_delay` (simulation truth) |
| Excluded | anything from the truth record — every feature exists for real Roman data |

## 4. Honest evaluation — two protocol rules

1. **Out-of-fold predictions, grouped by lens system.** 5-fold
   `GroupKFold` on `lens_index`: train on 4/5 of the *systems*, predict the
   held-out fifth, rotate. Every reported number comes from a model that
   never saw that system. (Tabular version of the "split by system, never
   by realization" rule.)
2. **Calibrated error bars, checked.** Two extra models trained with
   quantile loss (16% and 84%) bracket a per-system uncertainty band; the
   script reports how often the truth falls inside it (target: 68%).

Also reported: **permutation importance** — shuffle one feature, measure
the accuracy loss — showing which summaries actually carry the signal.

## 5. What it can and cannot learn

**Within reach** (corrections conditioned on context):
- systematic bias fixes — "high band scatter + faint second image →
  `dt_gp` overshoots by ~5%, subtract it";
- alias handling — "`multipeak` at low delay usually grabbed a correlation
  alias; shift by the typical peak spacing";
- error recalibration — "`dt_gp_err` is underestimated at z > 3 with two
  bands; inflate the band".

**Structurally out of reach:** anything absent from the 15 summaries. If
the GP fundamentally missed and nothing in the summaries hints at the true
answer, no tree ensemble recovers it. That ceiling is the *point* — see §6.

## 6. Strategic role: the bar the transformer must clear

The GBT sees 15 summary numbers; the transformer will see ~400 raw
photometric points. Their accuracy gap = the measured value of raw light
curves over summaries.

- **GBT ≈ robust fit accuracy (~1.2 d P68)** → summaries nearly saturate
  the delay information; the transformer's justification becomes
  microlensing + joint inference, not delays. Prioritize the microlensing
  sims over network engineering.
- **GBT plateaus well above** → quantified proof the delay signal lives in
  the raw curves; the transformer has a concrete number to beat.

Either way, every future results table gets its "simple method" row — the
one referees ask for.

## 7. Outputs

- Metrics table on stdout (same columns as `compare_fit_modes.py`:
  median, P68 |res|, %<2 d, %<5 d) for raw GP vs GBT, overall and split by
  GP quality flag.
- Quantile-band coverage (calibration) and median band half-width.
- Top permutation importances.
- Out-of-fold predictions → `scripts/gbt_baseline_predictions.ecsv`
  (per system: `gbt_pred`, `gbt_q16`, `gbt_q84`, `residual`, quality).

## 8. Results (2026-07-06; residual target, early stopping)

| method | n | median | P68 \|res\| | <2 d | <5 d |
|---|---|---|---|---|---|
| raw GP | 1062 | −0.11 d | 3.86 d | 55% | 73% |
| GBT | 1062 | +0.01 d | 5.24 d | 46% | 67% |
| raw GP (good) | 670 | −0.02 d | 1.59 d | 74% | 91% |
| GBT (good) | 670 | −0.02 d | 2.30 d | 64% | 85% |
| raw GP (not good) | 392 | −2.42 d | 163.8 d | 22% | 42% |
| GBT (not good) | 392 | +0.26 d | 47.7 d | 16% | 36% |

Calibration: truth inside the GBT's [16%, 84%] band 60% of the time
(target 68% — mildly overconfident). Top features: `dt_gp_err`, `dt_gp`,
`quality=fail`, `band_scatter`, `snr_img`.

**Verdict:**

1. **Summary features are saturated on trusted systems.** Where the GP
   flags itself `good`, no correction layer on the 15 summaries beats the
   raw estimate (2.30 vs 1.59 d — the trees only add variance). Tried and
   ruled out: absolute-target (2× worse — see fixes.md), residual target,
   early stopping.
2. **The transformer's delay case is therefore precise:** any improvement
   over 1.59 d P68 must come from the raw light curves, not summaries;
   the gap to the robust fit (1.19 d) is the available delay headroom. Its
   primary payload remains microlensing + joint inference, as designed.
3. **Where the GBT does add value:** the untrusted subset — P68 drops from
   164 d to 48 d (3.4×). Not measurement-grade, but potentially useful for
   routing/prioritizing fallback systems. Side product, not the main
   course.
4. **Method lesson (also in fixes.md):** with a strong baseline estimate
   in hand, predict *corrections to it*, never the absolute quantity —
   piecewise-constant trees pay a heavy tax re-learning identity.
