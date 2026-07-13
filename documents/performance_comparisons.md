# Performance Comparisons — Time-Delay Methods

Every method as a cost-vs-accuracy ladder, ours and the literature's, with
conditions attached. Companion to `pipeline_architecture.md` §5.4 (short
form + paths-to-beat live there). **Last updated 2026-07-13** (v4 final;
truncation eval preliminary — crop_v4 mid-training rerun pending).
P68 = 68th percentile of |Δt residual|.

## Our classical stack (measured)

| stage | what it is | P68 | cost / system | conditions |
|---|---|---|---|---|
| GP cross-correlation (Stage 1) | Matérn-3/2 GP per image + inverse-variance cross-correlation; no SN model, no sampling | **1.91 d** | 4 s | spec HLTDS deep sims, val hints vs truth; good-flag subset 1.29 d |
| GP → SNTD fast (Stage 3) | nested sampling of SALT model inside GP windows | **1.48 d** | ~4.5 min | 185-system suite, PRE-spec sims (optimistic); good subset 1.37 d |
| SNTD fast, unprimed | self-recentered windows | 1.88 d | ~4.3 min | same suite — priming beats it on accuracy AND reliability |
| SNTD robust | wide-bounds nested sampling | **1.14 d** | ~55 min | same suite; the accuracy ceiling of our pipeline |

## Our transformer stack (v4 = deep31k_full_v4 final)

Structural note: the transformer CANNOT run without Stage 1 — the GP
supplies its phase zero-point and (since v3) the anchor its delay output
corrects. Every row below includes the GP's 4 s.

| stage | what it is | P68 | cost / system | status |
|---|---|---|---|---|
| GP → transformer | anchored correction + calibrated σ per system | **2.07 d** all / **1.91 d** hinted (= GP tie, unbiased) | 4 s + 18 ms (CPU) | measured |
| GP → transformer, hints withheld | curves + anchor only (GP-failure drill) | 2.93 d (v2 arch); v4 unmeasured (weaker fallback branch by design) | same | partial |
| transformer, no GP at all | — | impossible as built (no clock zero-point) | — | structural |
| GP → transformer → SNTD fast | sampling inside transformer dt ± 4σ windows + theta/dust priors (the extra the GP can't give) | expected ≈ 1.48 d; unknown if priors tighten posteriors / cut fit time | ~4.5 min | **NOT RUN** — queued (~4 h for 50 systems) |

Where the transformer adds value at equal accuracy: calibrated per-system
σ (75% cov before calibration pass), quality-aware error bars, answers on
GP-fail systems (poor: ~200 d), joint mag-ratio/SN-param/micro heads.
Where it adds none (measured, 07-13): tier-1 delay accuracy — tie at every
quantile, every GP-quality class, and the outlier tail.

## Truncation evaluation (07-13, `scripts/eval_truncated.py`)

889 val systems clipped to a random 365 d observer window (seeded,
deterministic); GP re-run on the clipped curves (0 hard failures);
models scored with the clipped-GP hints — the realistic inference chain.

| category | systems | GP (clipped) | full-trained | crop-trained* |
|---|---|---|---|---|
| mild — all peaks kept | 591 | 1.96 d | **1.83 d** | 1.85 d |
| peak — a non-ref peak lost | 22 | 12.3 d | 20.9 d | 14.5 d (tiny n) |
| severe — ref peak lost | 255 | 180 d | 183 d | 180 d |

*crop-trained checkpoint was mid-training (epoch ~34/90); rerun pending.

Verdict: the window either preserves the delay information (everyone
~2 d; transformer's first small accuracy edge, ~7% over the degraded
clipped-GP hints) or destroys it (ref peak lost → ~180 d for ALL
template-free methods; crop training rescues nothing because there is
nothing to rescue). Open follow-up: can Stage-3 TEMPLATE fits pin a peak
from tail-only data where template-free methods cannot? Untested.

## Literature

| method | what it is | precision | cost / system | conditions |
|---|---|---|---|---|
| HOLISMOKES XII LSTM-FCNN (2024, arXiv:2403.08029) | LSTM+FCNN on follow-up light curves | **0.7 d**, bias-free | ~ms inference; requires TRIGGERED FOLLOW-UP (i-band every 1–3 d, 24.5 mag — telescope time is the real cost) | LSST lensed SNe Ia mocks + follow-up; NOT survey cadence |
| HOLISMOKES VII Random Forest (2022) | RF on same follow-up curves | 1.4 d | same follow-up requirement | same |
| Pierel+ 2021 (ApJ 908,190) | SNTD-style template fitting forecast | **~2 d** (SN Ia) | ~minutes (fitting) | Roman HLTDS survey cadence — our regime; our Stage-3 rows beat it |
| GausSN (2024, arXiv:2311.17997) | Bayesian GP joint-fit (delay a parameter) | 43.6% of delays < 5% frac err (ours: 50.7%) | ~minutes (MCMC) | their own Roman mocks — cross-sim caveat; rigorous test = run GausSN on OUR sims |
| PyCS3 / COSMOGRAIL (quasars) | spline/regression-difference curve shifting | ~1–2 d on multi-year quasar curves | ~minutes | different source class: years-long baselines, stochastic variability |
| SN Refsdal (Kelly+ 2023) | measured, 5 images, cluster lens | ±5.6 d on 376 d = **1.5%** | years of HST monitoring + analysis | real data; long delay makes the fraction small |
| SN H0pe (Pierel+ 2024) | measured, 3 images, JWST | ±4–10 d ≈ 8% | 3 JWST epochs + full analysis | real data; what practice currently achieves |

## Read of the whole board

1. **Survey-cadence information floor ≈ 1–2 d.** Template fits, GPs, and
   neural nets — ours and published — all land there. No method extracts
   what 5–10 d sampling doesn't record.
2. **Sub-day published numbers buy it with data, not method** (HOLISMOKES
   follow-up cadence). The equivalent lever for us: a follow-up
   simulation mode in the builder, then re-train/evaluate.
3. **Best accuracy per CPU-hour**: GP → SNTD fast (1.48 d @ 4.5 min).
   **Best accuracy**: SNTD robust (1.14 d @ 55 min). **Best per
   millisecond**: GP alone / GP+transformer (1.9 d).
4. **Open cells that could still move the board**: transformer-primed
   SNTD (theta/dust priors); Stage-3 template fits on severely truncated
   curves (can a shape prior pin a missing peak?); hints-off v4; GausSN
   on our sims; tier-2 microlensing (no published method attempts joint
   delay+micro — the differentiator if it works; carries nearly all the
   transformer's remaining case after the tier-1 tie and truncation
   null).

## Caveats ledger

- Our Stage-3 rows: pre-spec sims (flat 5 d cadence, deeper depths) —
  re-benchmark on spec sims pending; expect them to degrade somewhat.
- Fractional-error comparisons across simulation suites are indicative
  only (delay-distribution dependence).
- v4 cost figures are CPU (8 threads); GPU would cut the 18 ms, not the
  4 s GP prerequisite.
- Real-data rows (Refsdal, H0pe) benchmark end-to-end practice including
  systematics we don't simulate yet (microlensing, cluster environments).
