# Fixes Log

Major bugs and their fixes, newest first. One line each — details live in
git history. Add a row when a bug costs > 1 hour or changes results.

| Date | Problem | Root cause | Fix | Files |
|---|---|---|---|---|
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
