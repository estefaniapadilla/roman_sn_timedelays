# Fixes Log

Major bugs and their fixes, newest first. One line each — details live in
git history. Add a row when a bug costs > 1 hour or changes results.

## Pending (known, NOT yet applied)

(none)

**(b) RETRACTED 07-10** — suspected GP-hint slot misalignment (ref image at
slot 0 shifting hints vs targets) is **not a bug**: verified on all 8,286
tier1_deep records that `per_image` never contains the ref and its key order
matches the target slots exactly. Root of the false alarm: a wrong code
comment in `tokenize.py` ("slot 1 is the reference"), trusted over the data.
Comment corrected + defensive ref skip added (bit-identical on 500 systems —
no retrain). Lesson: same as the t0 bug — check the data, not the comment.

| Date | Problem | Root cause | Fix | Files |
|---|---|---|---|---|
| 07-12 | Crop augmentation crashed (`ValueError: high - low < 0` in `_crop_window`'s `uniform`) killing deep31k_crop_v4 at epoch 0 | Window-start range `[lo−0.3W, hi−0.7W]` is invalid when the curve spans < 0.4·W = 146 d; old 10k population had zero such systems (min z_source too high), 31k build sampled the rare low-z_source tail — latent since the feature was built | Short-span guard: skip the crop (return uncropped, kept_frac 1.0). Verified: all 21,531 train examples sweep clean; guard hits 15 (0.07%), span min 130 d. Lesson: sweep augmentations over the WHOLE dataset, tails emerge at scale | `ml_data.py` |
| 07-10 | Residual delay head, two failed forms before the working one: (i) ONE output unit for both regimes → biased compromise (+25 d by epoch 1: "output 0" for 80% hinted vs "output full delay" for 20% dropped-hint conflict); (ii) values gated but ONE shared σ → stalemate (37 no-hint slots' huge misses hold σ at ~100 d for everyone; at σ=100 d the miss/σ² pull is too weak, correction head wanders, +3 d growing bias by epoch 4) | Heteroscedastic-NLL regime mixing — fix (d) scenario | Gate value AND σ by hint presence: hinted = (hint + corr, σ init 10 d), no-hint = (absolute, σ init 100 d), `torch.where` routes gradients per branch. Untrained floor = GP accuracy exactly; val_dt starts ≈ +1.25 (GP-error tail under tight σ — the training signal), goes negative as σ splits by quality flag | `transformer.py` |
| 07-10 | `best.pt` + early stop keyed on val_total, dominated by the trivially-learnable heads (micro zeros, theta, dust) — could keep the wrong epoch's weights for delays and mis-time early stopping | Sum-of-heads selection metric | Select/early-stop on val_dt. No retrain: criterion doesn't affect gradients; in v2 it cost only 0.002 (ep 57 vs 56) — takes effect in the next run | `train_transformer.py` |
| 07-09 | Transformer delay head useless (val P68 ≈ 40 d vs 1.9 d from just copying the GP hint; predict-zero = 51 d) in deep_full_v1/deep_crop_v1 | dt targets + dt_gp(_err) hints fed as raw days (pop std 129 d) → heteroscedastic NLL inflates σ (p90 = 234 d), mean-gradient ∝ 1/σ² stalls; predictions collapsed to std 12 d | `DT_SCALE = 100`: targets and hints stored as days/100, metrics ×100 back to days. Retrain = v2 runs | `tokenize.py`, `ml_data.py`, `train_transformer.py` |
| 07-07 | **BayeSN sim (Path B) evaluated the SED ~60,000 d after peak** — every flux in the tier1 training set (384 examples) is wild model extrapolation, some bands 10³⁰× too bright; smoke + tier1_v1 transformer runs trained on garbage. Benchmarks unaffected (Path A/lenstronomy) | `span_lo = t0 + model.mintime()` double-counts t0 (sncosmo min/maxtime already include t0); the phase cut compared relative phase to absolute time, masking the crash | Span = model time range directly; phase cut on `mjd − dt` vs model range. Verified: peak mags 24–26, image peaks at t0+dt, GP recovers delays | `simulate.py` |
| 07-07 | Sims too optimistic vs HLTDS spec: every filter at 5 d — HLTDS spec is one anchor filter at 5 d + rest at 10 d (per tier) → existing benchmarks/training set have ~2× too many points in the 10-day filters (optimistic) | Simulator had a single global cadence; per-filter CCS spec not yet encoded | `SURVEY_CADENCE` + `SURVEY_EXPTIME` + `SURVEY_DEPTH_5SIG` in `simulate.py` (depths from CCS exposure times × WFI 1-hr sensitivities, m5 − 1.25·log₁₀(3600/t)); `simulate_photometry` takes per-band cadences; builder uses per-tier spec by default | `simulate.py`, `build_training_set.py` |
| 07-06 | **Verdict: BayeSN two-stage INFEASIBLE** — even with every mitigation (GP-primed windows, log-uniform amplitude, 200k-call cap), one system took 5.4 h, both stages hit the cap, and the delay came back pinned at the −30 d window bound (true +9.78 d) with zero-width errors | Plateau pathology (below) survives all repo-side mitigations at these SNRs | Abandon series/color route; forward path = BayeSN SED via SNTD **parallel** method (unbuilt) | (verdict — docs only) |
| 07-06 | GBT baseline scored 2× *worse* than the raw GP it was built on | Target was the absolute delay: piecewise-constant trees waste all capacity re-learning the trivial "prediction ≈ dt_gp" identity | Predict the residual (true − dt_gp) and add it back — "predict 0" then equals raw-GP accuracy | `gbt_baseline.py` |
| 07-06 | Historical benchmark misattributed | `delay_benchmark.ecsv` is from the **SALT** driver, not BayeSN — BayeSN two-stage has *never* been seen to converge on this population | Treat BayeSN viability as an open feasibility question | (docs only) |
| 07-06 | BayeSN fits grind forever (13 h) or return bound-pinned garbage with fake zero-width errors | Likelihood plateaus (SNTD rounds phases to 0.1 d) + linear-uniform amplitude prior over 90,000× range + blind ±40 d delay windows | `maxcall` cap + `converged` flag in benchmark; log-uniform amplitude prior; GP-primed delay windows; color-stage crash caught → series fallback | `bayesn_wrapper.py`, `run_bayesn_population.py` |
| 07-05 | GP: confident wrong delay (−179 d vs +25 d true) | Flat zero-flux baselines correlate spuriously at extreme lags | Only score lags where bright parts (>30% of peak) of both curves overlap | `crosscorr.py` |
| 07-05 | GP: 23 s/system (too slow) | sklearn `sample_y` does a full SVD per call | Manual Cholesky posterior draws → 4 s/system | `crosscorr.py` |
| 07-05 | GP: broadcast crash at extreme lags | `slice(0, n−k)` wraps around when k > n | Skip lags near/beyond grid length | `crosscorr.py` |
| 07-05 | GP: clean systems flagged `broad`, never `good` | Flag used correlation-peak width — but SN curves always give wide peaks | Classify on GP-draw scatter + rival peaks; width dropped | `crosscorr.py` |
| pre | SALT fit at z=0 against z≈1–4 data | SNTD never reads `misn.zs` automatically | Pass `constants={"z": z_source}` (commit b8cb628) | `sntd_wrapper.py` |
| pre | nestle degenerate-weights crash | Same plateau family as above | `_patch_nestle_weights()` runtime workaround | `sntd_wrapper.py` |
| pre | KeyError extracting delays | SNTD version differences in output format | Tolerant dict/list handling (commit 679fa35) | `sntd_wrapper.py` |

## Notes

- **Plateau mechanics** (07-06): SNTD's 0.1-day phase rounding makes the
  likelihood a staircase; nested sampling needs strictly-better draws, ties
  get rejected → ~2% acceptance, or all-tie → `weights sum to zero` crash.
  Fix constraint: no changes to installed packages — repo-side mitigations
  only (smaller search volume, caps, flags).
- **Key numbers** (07-06): BayeSN likelihood 33 ms/call vs SALT 12 ms;
  log-uniform amplitude alone raised sampler acceptance 5×; still hit the
  50k cap on the test lens.
- **Feasibility test** (07-06, behind the verdict row): lens 2, zS=3.62,
  2 bands, true dt +9.78 d; npoints=500, maxcall=200k/stage, GP hints ON.
  Series 38 min / 200,082 calls (capped); color 287 min / 200,009 calls
  (capped); dt fit −30.00 [−30.00, −30.00] (bound-pinned), mu fit 2.496
  pinned (true 0.532). 5.4 h → garbage. Two-stage route closed.
