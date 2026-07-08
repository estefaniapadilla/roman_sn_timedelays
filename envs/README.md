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

## roman_ml usage note

Always run via activation (`conda activate roman_ml` or
`conda run -n roman_ml ...`) — never call its `bin/python` directly.
The env sets `LD_LIBRARY_PATH` on activation so pip-installed torch loads
the env's libstdc++ instead of the (older) system one; bypassing
activation resurfaces `CXXABI_1.3.15 not found` on `import torch` +
scipy. Also: create envs with `--solver=libmamba` (the classic solver
took hours on this machine).

## Rules

- Never `pip/conda install` into `sntd_bayesn` — its numpy<2 / sntd / slsim
  solve is fragile. New dependencies go in `roman_ml` (or a new env).
- After any approved change to an env, re-export its lock file:
  `conda env export -n <env> > envs/<env>.lock.yml`
- `roman_td.tokenize` and `roman_td.paths` are pure numpy by design and
  importable from either env; the fitting modules (`sntd_wrapper`,
  `bayesn_wrapper`, `simulate`, `crosscorr`) require `sntd_bayesn`.
