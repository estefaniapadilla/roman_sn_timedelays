# Environments

Two purpose-built conda envs; they communicate only through files on disk
(`outputs/training/*`), never in one process — so neither can break the other.

| env | job | runs | spec |
|---|---|---|---|
| `sntd_bayesn` | fitting + simulation (FROZEN — do not install into it) | `run_sntd_population.py`, `run_bayesn_population.py`, `run_gp_population.py`, `build_training_set.py`, `compare_fit_modes.py`, `gbt_baseline.py` | `sntd_bayesn.yml` (portable) / `sntd_bayesn.lock.yml` (exact clone) |
| `roman_ml` | transformer training + evaluation | `train_transformer.py` (future) | `roman_ml.yml` |

## Recreate

```bash
conda env create -f envs/sntd_bayesn.yml
git clone https://github.com/bayesn/bayesn && pip install -e ./bayesn   # 0.4.1 — not on PyPI
conda env create -f envs/roman_ml.yml
```

## Rules

- Never `pip/conda install` into `sntd_bayesn` — its numpy<2 / sntd / slsim
  solve is fragile. New dependencies go in `roman_ml` (or a new env).
- After any approved change to an env, re-export its lock file:
  `conda env export -n <env> > envs/<env>.lock.yml`
- `roman_td.tokenize` and `roman_td.paths` are pure numpy by design and
  importable from either env; the fitting modules (`sntd_wrapper`,
  `bayesn_wrapper`, `simulate`, `crosscorr`) require `sntd_bayesn`.
