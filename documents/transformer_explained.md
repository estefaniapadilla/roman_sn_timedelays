# The Transformer (Stage 2), Explained

A detailed guide to the planned Stage-2 neural network — what it is for,
where it sits in the pipeline, exactly what goes in, what happens inside,
the training and serving infrastructure, and what comes out. Companion to
`pipeline_architecture.md` §3 (the specification) and
`gp_crosscorr_explained.md` (Stage 1, which feeds it). Status: **[TO BUILD]**
— gated on the training-set builder (architecture §7 item 6), the
microlensing injection (item 7), and the gradient-boosted-tree baseline
(item 8), in that order.

---

## 1. What it is for, in one paragraph

The transformer is an *amortized* inference engine: after an expensive
one-time training on simulated systems with known truth, it answers in
~milliseconds per system what the physical fits answer in minutes-to-hours —
time delays, magnification ratios, SN parameters — each with a calibrated
uncertainty. Beyond speed, it has one scientifically novel job that the
current fits cannot do at all: **separating macro-lensing from
micro-lensing**, because it can learn the different fingerprints the two
effects leave on multi-band, multi-image light curves. It never replaces the
physical fit; it narrows the fit's search space further than the GP can, and
provides fast catalog-scale numbers plus microlensing flags.

## 2. Where it plugs in

```
Stage 0 (simulation) ────────────── training data + truth labels
        │  canonical photometry
        ▼
Stage 1 (GP cross-correlation) ──── dt_gp ± σ, flux ratio, quality flag,
        │                           per-band scatter, t_peak_ref
        ▼
┌──────────────────────────────┐
│ Stage 2 · TRANSFORMER        │  ~ms per system after training
└──────────┬───────────────────┘
           │  tight dt bounds, theta/A_V priors,
           │  mu ratios, microlensing flags + amplitudes
           ▼
Stage 3 (SNTD physical fit) ─────── publication-grade posteriors
```

Concretely, three plug-in points:

1. **Upstream (inputs):** it consumes the *same canonical photometry table*
   (architecture §6.1) the GP consumes — one row per (image, epoch, band) —
   plus the GP's §6.3 output as hint features, plus the redshifts.
2. **Downstream (Stage 3 priming):** its `dt_i ± σ` become per-image t0
   bounds in `measure_one()` (SALT path) or `dt_*` bounds in `fit_system()`
   (BayeSN path) — mechanically identical to the GP handoff, just tighter.
   Its `theta_hat ± σ` replaces the generic N(0,1) theta prior in the BayeSN
   series fit; its `A_V` estimate narrows the `hostebv` bounds.
3. **Sideways (catalog + flags):** for population science, its outputs *are*
   the fast catalog (delays, mu ratios for thousands of systems), and its
   microlensing head flags contaminated systems for the expensive robust
   treatment. Microlensing outputs do NOT feed Stage 3 (the fit model has no
   microlensing term to prime).

Both redshifts (`z_lens`, `z_source`) are guaranteed available for every
system — survey design — so they are exact inputs everywhere below, with no
missing-value machinery.

## 3. Why a transformer specifically

The input is an awkward object for classical ML: a *set* of photometric
points, irregularly spaced in time, spread across ≤6 bands and 2–4 images,
with a different count per system (typically 100–600 points). A dense
network needs fixed-size input; a CNN wants a regular grid; an RNN struggles
with irregular gaps and interleaved images.

A transformer consumes a variable-length set of tokens natively, and its
core operation — **self-attention**, where every token computes its
relevance to every other token — matches the physics one-to-one:

- "this point on image B's rise looks like that point on image A's rise,
  offset by Δt" — *attention across images at a time offset* → the delay;
- "image B is consistently this factor fainter at matched phases" →
  the magnification ratio;
- "image B's F087/F158 color drifts relative to image A's at matched
  phases" — a per-image, chromatic, time-varying deviation → microlensing.

The architecture doesn't have to be taught these comparisons exist; it has
to *discover* them from labeled examples, and attention is the right
inductive bias for pairwise-comparison discovery.

## 4. Inputs, precisely

### 4.1 The token sequence (one token = one photometric measurement)

```
token = [ phase_norm, flux_norm, fluxerr_norm,
          band_onehot(6), image_onehot(4), is_real ]     -> 13 features
```

Construction order matters; each step exists for a reason:

1. **`phase_norm`** = `(mjd − t_peak_ref) / (1 + z_source)`, where
   `t_peak_ref` is the GP peak time **of the reference image only, shared by
   every token in the system**.

   > **The one silent-failure trap (worth repeating everywhere):** if each
   > image were phased to *its own* peak, the delay would be subtracted out
   > of the input — the network would have nothing left to learn the delay
   > from and would parrot the GP hint. One zero-point per *system*: a
   > non-reference image's tokens must sit ≈ Δt/(1+z) away from the
   > reference's. That offset IS the signal.

   The (1+z) division puts every SN on the same intrinsic clock (a z=3.8 SN
   evolves 4.8× slower in observer days than a z=0 one), so the network
   spends no capacity learning time dilation. The *output* delay stays in
   observer-frame days — that is what carries H0 information.

2. **`flux_norm`** = flux ÷ (brightest detection across all images, bands,
   epochs of this system); `fluxerr_norm` same denominator. Per-system
   internal normalization preserves *ratios between images* (the
   magnification signal) while removing absolute scale.

   > **Never** normalize via a cosmological distance modulus: the distance
   > modulus folds in the magnification — one of the measurement targets —
   > and normalizing that way silently divides the signal out.

3. **`band_onehot`** over the six Roman filters (F062…F184), fixed order,
   zeros for filters absent in a survey tier. Essential for microlensing:
   the effect is chromatic, so the network must know which band each point
   is.
4. **`image_onehot`** over slots 1–4, assigned in the truth record's image
   order with the reference image always slot 1. Essential because
   microlensing is *per-image* while the SN and host dust are *shared* —
   the disentangling lever (§6).
5. **`is_real`** = 1 for real measurements, 0 for padding. Systems are
   padded to a fixed length `L_max` (the training set's 99th-percentile
   token count, ≈ 400–600 at Roman's 5-day cadence) and padding is masked
   out of attention.
6. **Non-detections are tokens too.** An observed epoch where image B shows
   nothing is data: it pins the delay's lower bound ("B had not risen yet").
   Flux near zero with its real error bar — not dropped.

### 4.2 Per-system scalar features (fused after pooling)

```
[ z_source, z_lens, n_images,
  dt_gp_slot2..4, dt_gp_err_slot2..4,          # GP hints, per image slot
  flux_ratio_slot2..4,
  gp_band_scatter, gp_quality_onehot(4) ]
```

- Redshifts enter here (not per-token): exact values, no jitter, no
  missing-value handling — the survey guarantees them.
- **GP-hint dropout:** during training, with probability ~0.2 all GP-derived
  features are zeroed and the quality one-hot forced to `fail`. This trains
  the network to solve the problem from tokens alone and treat the hint as a
  refinement — so a wrong GP number at inference (the ~9% heavy tail
  measured in the 1k benchmark) cannot fully steer the prediction.

## 5. What happens inside (the infrastructure of the model)

Target size ≈ **1–3 M parameters** — deliberately small. Sizing rule: do not
scale up until both the GBT baseline and this size are saturated.

```
tokens (L_max × 13)
  │  linear embed -> d_model = 128
  ▼
[CLS] token prepended                      # learned summary slot
  │
  ▼
Transformer encoder × 4–6 blocks:
  ├─ pre-norm LayerNorm
  ├─ multi-head self-attention (8 heads, d_head = 16), padding masked
  ├─ residual add
  ├─ pre-norm LayerNorm
  ├─ MLP 128 -> 512 -> 128 (GELU), dropout 0.1
  └─ residual add
  ▼
take [CLS] output (128)  ── concat ──  scalar features (~20)
  │
  ▼
fusion MLP: 148 -> 256 -> 256 (GELU)  =: h, the shared representation
  │
  ├─ delay head        : per slot i∈{2,3,4}: (dt_i, log σ_i)
  ├─ magnification head: per slot i∈{2,3,4}: (log μ_i/μ_ref, log σ)
  ├─ microlensing head : per image incl. ref: (A_micro, dA/dband, log σ each)
  └─ SN head           : (theta_hat, A_V_hat, log σ each)
```

Design notes:

- **No positional encoding of sequence order.** Token order is meaningless
  (a set, not a sequence); *time* enters through the `phase_norm` feature.
  This is a Set-Transformer-style treatment.
- **Fixed-size outputs, variable-size systems:** heads predict *per image
  slot relative to the reference*, with a validity mask zeroing loss for
  absent slots (a double uses slot 2 only; a quad uses 2–4). Never
  "per pair" — that's a variable-size output. This parameterization mirrors
  SNTD's series fit (`dt_img`, `mu_img/mu_ref`), so Stage-3 handoff is a
  direct substitution.
- **Every head is heteroscedastic:** it predicts a value *and* its own
  uncertainty (as log σ for positivity). Joint prediction from the shared
  `h` is deliberate — microlensing masquerades as color, color as dust; one
  representation lets each head's evidence constrain the others, which the
  sequential physical fits cannot do cleanly.

## 6. How it separates macro from micro (the novel capability)

The two effects leave different, learnable fingerprints:

| | macro-magnification | micro-lensing |
|---|---|---|
| wavelength | achromatic (same factor in all bands) | chromatic (SN's blue photosphere is smaller → more point-like → stronger micro) |
| time | constant | drifts over weeks (photosphere expands across the magnification map) |
| images | independent per image but *static* | independent per image and *varying* |

And the disentangling lever: **the SN itself and its host dust are shared by
all images** (same explosion, same host), so they imprint *identically* on
every image. Whatever chromatic, time-varying, *per-image* residual remains
after aligning the images in time and removing a flat per-image factor — is
microlensing. Attention across images at the learned delay offset is exactly
the mechanism that can compute this residual. The GP's Δt hint pre-aligns
the problem; the image one-hots tell the network which points share a
sightline.

This only works because the training data contains *labeled* examples with
known injected microlensing (tier-2 sims, architecture §1.3 Path C) — which
is why the microlensing injection gates the transformer build.

## 7. Training infrastructure

- **Data:** Stage-0 training set — canonical photometry + §6.2 truth JSON
  per system. Order 10⁴ systems minimum; the sims are cheap enough (seconds
  per system for the SALT-path simulator) to generate several noise/cadence
  realizations per lens. **Split by lens system, never by realization** —
  realizations of one system in both train and test is leakage.
- **Labels & loss:** per-head Gaussian negative log-likelihood
  `0.5·[(y−ŷ)²/σ̂² + log σ̂²]`, summed over heads, with per-target masks.
  In tiers 0–1 the micro targets are *zero, not masked* — the network must
  learn to report "no microlensing", not just to ignore it.
- **Curriculum:** first epochs on tiers 0–1 (learn the clean delay), then
  phase in tier 2 (micro), then 3 (milli, research-grade). Networks learn a
  hard disentangling task better when eased in.
- **Model-swap robustness test:** train on BayeSN-simulated, test on
  SALT-simulated systems (and vice versa). The degradation measures
  simulator-dependence — the central scientific risk — and is reported next
  to every accuracy number.
- **Calibration pass (mandatory, not optional):** after training, measure
  empirical coverage of the σ̂'s on the validation split (68%/95% targets)
  and fit per-head temperature scaling (or conformal offsets) as a frozen
  post-processing step. The GP's 1k benchmark set the bar here: 70% within
  1σ. If delay residuals show multimodality (peak aliasing), upgrade the
  delay head to a small mixture density network — decide from validation
  residuals, not in advance.
- **Compute:** at 1–3 M parameters and ~10⁴–10⁵ examples this trains on a
  single modest GPU in hours, or on CPU overnight — this is *not* a
  large-model project. PyTorch; AdamW; cosine schedule; early stopping on
  validation NLL. Everything seeds-fixed and config-logged so a training run
  is reproducible from its config file.
- **Repo layout:** `roman_td/tokenize.py` (table → tokens + scalars, reused
  identically at train and inference time — one code path, no train/serve
  skew), `roman_td/transformer.py` (model + heads), and
  `scripts/train_transformer.py` (driver writing checkpoints + a frozen
  calibration file + a training-config JSON).

## 8. Outputs, precisely (architecture §6.4)

Per system, all uncertainties *post-calibration*:

| field | per | meaning |
|---|---|---|
| `dt_i`, `dt_err_i` | image slot 2–4 | observer-frame delay vs reference, days |
| `log_mu_ratio_i`, err | image slot 2–4 | total magnification ratio vs reference |
| `micro_amp_i`, err | every image | microlensing amplitude (mag); ~0 = clean |
| `micro_slope_i`, err | every image | chromatic slope of the micro signal |
| `theta_hat`, `av_hat`, errs | system | SN shape + dust — become Stage-3 priors |
| `model_version`, `calibration_version` | system | provenance: which checkpoint + calibration produced this row |

Downstream use, restated: `dt_i ± 4σ` → Stage-3 t0/dt bounds;
`theta_hat ± σ` → BayeSN theta prior; `micro_amp` → *flag only* (routing
systems to robust treatment and to microlensing studies) — quoted as a
measurement only after validation against Stage 3 and matched-pair ablations
(tier-2 vs tier-1 twins).

## 9. Honest limits and how they're contained

- **It only knows what the simulator taught it.** If real microlensing or SN
  diversity differs from the sims, the network can be confidently wrong in
  ways a physical fit would not be. Containment: model-swap tests, the
  calibration pass, Stage 3 always re-measures anything that matters, and
  the residual between Stage 2 and Stage 3 on overlapping systems is itself
  a monitored diagnostic.
- **The GBT baseline gates it.** If gradient-boosted trees on GP summary
  features already meet the delay-accuracy bar (they can be built *today*
  from `gp_benchmark.ecsv`), the transformer's unique value is microlensing
  and joint inference — worth knowing before, not after, the build.
- **Microlensing labels are only as real as the injection.** An ad-hoc
  injected signal (e.g. smooth random walks) trains a network to find
  ad-hoc signals. The injection must come from magnification maps convolved
  with an expanding, wavelength-dependent photosphere, or the micro head's
  output is decorative.
- **It is the accelerant, not the foundation.** Stages 0, 1, 3 constitute a
  working, publishable pipeline with no ML. The transformer's failure would
  cost speed and the micro capability — not the delays.
