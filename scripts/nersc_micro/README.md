# NERSC micro light-curve extraction

Extract 4-tier light curves from Sharma's precomputed lensed-SNe dataset
(`run_002`: 20 shards, 26,411 lenses, 241 GB). CPU-only — magmaps are cached
inside each `Lens`. Output: ~2–3 GB total, transferred back for training.

## Config (baked into extract_lightcurves.py)

| Setting | Value | Why |
|---|---|---|
| Bands | F062 F087 F106 F129 F158 F184 | HLTDS wide ∪ deep |
| Time grid | uniform 2 d, ≤600 epochs, per-lens window | dense truth; cadence/noise applied later by `build_training_set.py` |
| Tiers stored | intrinsic, macro, micro curves | per band |
| Millilensing | per-image constant factors only | achromatic; tier 4 = micro − 2.5·log10(f) |
| Labels | Δt, μ_macro, κ, γ1, γ2, κ★, z_l, z_s, θ_E | training targets + diagnostics |

## Run steps (on Perlmutter)

```bash
# 1. Copy dataset to own scratch (collab's copy may be purged; so may yours — scratch purges after ~8 weeks idle)
cp -r /pscratch/sd/s/sharma/lensed_sne_magmaps/run_002 $SCRATCH/lensed_sne_magmaps/

# 2. One-time env (login node)
bash setup_env.sh

# 3. Smoke test: 10 lenses of shard 00 (login or interactive node)
source $SCRATCH/envs/slsim_micro/bin/activate
python extract_lightcurves.py \
    --run-dir $SCRATCH/lensed_sne_magmaps/run_002 \
    --output-dir $SCRATCH/micro_lightcurves/smoke \
    --shard-id 0 --max-lenses 10 --workers 4
# → check lenses/hr in output; 12 h × 32 workers must cover ~1300 lenses/shard

# 4. Full array (adjust --time if smoke rate demands)
mkdir -p logs
sbatch submit_extract.slurm

# 5. Transfer results back (from STScI side)
scp "perlmutter.nersc.gov:\$SCRATCH/micro_lightcurves/run_002/*" \
    /home/epadill/time_delays/outputs/micro_lightcurves/run_002/
```

## Notes
- `--account=m1727` (CPU) taken from sharma's GPU script (`m1727_g`); fix if
  your allocation differs (`iris` shows your accounts).
- Reading a shard requires latest slsim to unpickle `Lens` objects; if
  unpickling fails with a ModuleNotFoundError, `pip install` the named package
  into the env (CPU mode is fine — no map generation happens here).
- Failures are per-lens, recorded in `micro_lc_set_NN.json`, never fatal.
- Rerun a failed task with the same array index; use `--overwrite` to redo.
