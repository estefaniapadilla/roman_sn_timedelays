# How the Transformer Works and Learns

Companion to `transformer_explained.md` (the design spec: why a transformer,
token layout, outputs). This document explains the machine itself, for a
reader new to neural networks: what goes in, what one forward pass computes,
what the loss function measures, and how the optimizer turns wrong answers
into better weights. Everything here describes the code as built —
`roman_td/transformer.py`, `roman_td/ml_data.py`,
`scripts/train_transformer.py` — with the actual numbers we run.

---

## 1. The job, in one sentence

Given the photometry of one lensed SN system (a few hundred flux
measurements across bands and images) plus a handful of scalar facts
(redshifts, GP hints), output the time delays, magnification ratios,
microlensing amplitudes, and SN parameters — **each with its own error
bar** — in about a millisecond.

## 2. What goes in

Two arrays per system (built by `roman_td/tokenize.py`, same code at
training and inference):

- **`tokens` (512 × 14):** one row per photometric measurement:
  `[phase, flux, fluxerr, band one-hot(6), image one-hot(4), is_real]`.
  Phase is rest-frame days from the reference image's GP peak (ONE
  zero-point per system — the inter-image offset IS the delay signal);
  flux is normalized by the system's brightest point (preserving the
  magnification ratios). Systems have ~100–450 real points; the rest of
  the 512 rows are zero-padding, flagged by a mask.
- **`scalars` (17):** `z_source`, `z_lens`, `n_images`, the GP's delay
  estimates/errors/flux ratios per image slot, GP band scatter, and the GP
  quality flag as a one-hot.

And, during training only, the **targets** from the truth JSON: true
delays, true log magnification ratios, microlensing amplitude/slope
(all zero in tier 1), theta, host E(B−V) — with masks marking which
image slots actually exist.

## 3. What one forward pass computes

"Forward pass" = pushing one batch of inputs through the network to get
predictions. Step by step (`LensedSNTransformer.forward`):

1. **Embedding.** Each 14-number token is multiplied by a learned 14×128
   matrix, turning it into a 128-number vector. Think of it as re-expressing
   each measurement in a richer internal vocabulary the network gets to
   invent.

2. **The CLS token.** A learned 128-vector (the same for every system) is
   prepended as token 0. It carries no data; it is an empty slot the network
   will fill with a *summary of the whole system* as the layers proceed —
   like a rapporteur sitting in on a meeting.

3. **Four encoder layers.** Each layer does two things to every token:

   - **Self-attention (8 heads).** Every token computes a relevance score
     against every other token (padding excluded via the mask) and updates
     itself with a weighted average of what the relevant tokens carry.
     This is the operation that can express "this point on image b's rise
     matches that point on image a's rise, 12 days earlier" — the delay —
     without being told to look for it. The 8 "heads" are 8 independent
     relevance patterns computed in parallel (one might track same-band
     comparisons, another cross-image alignment; the network decides).
   - **A small per-token MLP** (128 → 512 → 128, GELU nonlinearity)
     that transforms each token individually — the "thinking" step between
     rounds of "communication."

   Both steps use *pre-norm residual connections*: each block adds its
   output onto its input rather than replacing it, which keeps gradients
   well-behaved in deep stacks. Token order is never encoded — the input
   is a set, and time lives in the phase feature.

4. **Pooling + fusion.** After layer 4, the CLS token's 128 numbers are
   the system summary. They are concatenated with the 17 scalars
   (145 numbers) and passed through the fusion MLP (145 → 256 → 256).

5. **Four heads.** Small linear layers map the 256-number representation
   to the outputs. Every quantity comes out as a **pair (value, log σ)**:
   the prediction and the network's own claimed uncertainty. Predicting
   log σ rather than σ guarantees positivity.

Total: 0.91 M learned parameters. Deliberately small — rule of the
project: don't scale up until this size and the GBT baseline are saturated.

## 4. The loss function: Gaussian negative log-likelihood

The loss is the single number that defines "wrong." Ours (in `loss_fn`)
is, per target:

```
loss = 0.5 · ((y_true − y_pred) / σ)²  +  log σ
```

This is the negative logarithm of a Gaussian probability — "how surprised
is the network's own probability statement by the truth?" — and both terms
matter:

- The **first term** punishes misses, but scaled by the claimed σ: missing
  by 5 days with σ = 1 day is catastrophic; missing by 5 with σ = 10 is
  mild.
- The **second term** (`log σ`) punishes hedging: claiming huge σ on
  everything would zero the first term, so this term charges rent on
  uncertainty.

The optimum is honesty: σ as small as the evidence allows and no smaller.
This is how the error bars are *learned* rather than bolted on afterwards.
(`log σ` is clamped to [−5, 5] so a wild early prediction cannot produce
infinities.)

Per-head details:

| head | target | masked when |
|---|---|---|
| delays `dt` | true delay per non-ref slot, days | slot has no image, or (crop mode) image left with <3 points |
| magnifications `logmu` | log(μ_i/μ_ref) | same mask as dt |
| micro amp/slope | 0.0 in tier 1 | never masked — "no microlensing" must be *reported*, not skipped |
| SN params | theta, host E(B−V) | never |

The masks matter: a double system has one real delay slot of three; the
other two contribute nothing to the loss (division by the mask count, not
by 3). The total loss is the plain sum of the six per-head means — no
hand-tuned weights between heads (first thing to revisit if one head
dominates training).

## 5. How learning actually happens

One **training step** (one batch of 64 systems):

1. Forward pass → predictions → loss (one number).
2. **Backpropagation:** `loss.backward()` computes, for every one of the
   0.91 M parameters, the partial derivative ∂loss/∂parameter — "if I
   nudged this weight up a hair, would the loss rise or fall, and how
   steeply?" This is not an approximation; it is calculus applied through
   the whole computation graph, automated by PyTorch.
3. **Gradient clipping:** if the overall gradient vector is longer than
   1.0, rescale it. Insurance against a single pathological batch throwing
   the weights into a bad region.
4. **Optimizer step:** move every parameter a small distance in the
   direction that lowers the loss.

Repeat over ~104 batches = one **epoch** (one full pass over the 6,657
training examples, reshuffled each time). Because the augmentations
(window crop, GP-hint dropout) are re-randomized per epoch, the network
never sees exactly the same batch twice.

## 6. The optimizer: AdamW, with a cosine schedule

Plain gradient descent uses one global step size, which is fragile. We use
**AdamW** (`lr = 3e-4`, `weight_decay = 1e-4`), the field's default for
transformers, which adds three refinements:

- **Momentum:** each step blends in the running average of past gradients,
  smoothing out batch-to-batch noise (a heavy ball rolling downhill, not a
  drunkard's walk).
- **Per-parameter adaptive step sizes:** parameters whose gradients are
  consistently large get smaller steps, rarely-updated ones get larger
  steps. Every one of the 0.91 M weights effectively has its own learning
  rate.
- **Decoupled weight decay (the "W"):** every step, weights are pulled
  slightly toward zero — a gentle prior that small weights generalize
  better — applied separately from the gradient so it interacts correctly
  with the adaptive steps.

On top sits a **cosine annealing schedule**: the learning rate starts at
3e-4 and decays along a cosine curve to ~0 at the final epoch. Big steps
early (find the right valley), tiny steps late (settle into its floor).
Consequence: `--epochs` is not a mere cap — the schedule paces itself to
finish exactly there, so a 60-epoch run is a complete training, not a
truncated 200-epoch one.

## 7. Validation, early stopping, checkpoints

After every epoch the model runs on the **validation split** (879 held-out
systems — held out *by lens system*, so no noise-realization of a training
system can leak in) with augmentations off and no weight updates. That
yields `val NLL`, the honest score.

- The checkpoint with the lowest val NLL so far is saved to `best.pt` —
  we keep the best model, not the last.
- If val NLL hasn't improved for 20 epochs (`--patience 20`), training
  stops: the model is now memorizing training data, not learning.
  The discarded toy run showed the textbook signature — train loss falling
  4.0 → 1.0 while val loss climbed 8 → 20 after epoch 14. That is
  **overfitting**, and with 25× more data it should set in far later.
- Every run logs to `outputs/models/<name>/`: `best.pt` (weights),
  `config.json` (every hyperparameter + data fingerprint — the run is
  reproducible from this file), `history.csv` (per-epoch losses).

After training, the driver reloads `best.pt` and reports delay metrics in
benchmark form: median residual, P68 |residual|, % within 2 d, and the
fraction of residuals inside the claimed 1σ (calibration check — target
68%; the toy run said 100%, i.e. σ's massively over-wide). The bar to
beat: **GP 1.59 d P68 on good systems** (to be re-measured on the
spec-realistic sims).

## 8. Current hyperparameters at a glance

| knob | value | where |
|---|---|---|
| d_model / heads / layers / d_ff | 128 / 8 / 4 / 512 | `transformer.py` |
| fusion width | 256 | `transformer.py` |
| dropout | 0.1 | `transformer.py` |
| parameters | 0.91 M | — |
| batch size | 64 | `--batch` |
| learning rate | 3e-4, cosine → 0 | `--lr` |
| weight decay | 1e-4 | `train_transformer.py` |
| grad clip | 1.0 | `train_transformer.py` |
| epochs / patience | 60 / 20 (current runs) | `--epochs`, `--patience` |
| GP-hint dropout | 0.2 | `--gp_dropout` |
| window crop | p=0.7, 365 d (crop runs) | `--crop_prob`, `--crop_window` |
| l_max (token padding) | 512 | `--l_max` |
| seeds | torch + numpy fixed (42) | `--seed` |

## 9. Honest limits of the learning setup

- **The loss weights all heads equally.** If delays stall while micro
  targets (all zeros in tier 1) train easily, per-head weights are the
  first knob.
- **Gaussian error bars.** If delay residuals turn out multimodal (peak
  aliasing), a single (value, σ) cannot express "either +12 or +22 days";
  the planned upgrade is a small mixture density head.
- **Calibration is checked, not guaranteed.** The σ's minimize NLL on
  *training-distribution* data; the mandatory post-training calibration
  pass (temperature scaling on validation) corrects residual over/under-
  confidence before any number is quoted.
- **It learns the simulator, not the universe.** Model-swap tests
  (train BayeSN-sim / test SALT-sim) measure that gap; Stage 3 re-measures
  anything that matters.
