# Lensed SN Ia Time-Delay Pipeline — Design Document

A staged system for measuring gravitational time delays (and macro/micro-lensing)
of lensed Type Ia supernovae in the Roman High-Latitude Time-Domain Survey.

The goal: keep the accuracy and physical rigor of the current SNTD/BayeSN fits
while getting orders of magnitude more speed, and add macro- and micro-lensing
information that neither current pipeline produces.

---

## 1. Background — what the current pieces actually do

Before the new design, here is a precise account of the existing components, since
the whole design builds on understanding where their time goes.

### 1.1 SALT3 / SALT2-extended — the empirical SN model

SALT is a *data-driven* spectral template for SN Ia, trained on thousands of real
supernovae rather than simulated from explosion physics. It predicts flux at
rest-frame phase `p` and wavelength `lambda` as:

```
flux(p, lambda) = x0 * [ M0(p, lambda) + x1 * M1(p, lambda) ] * exp(c * CL(lambda))
```

- `x0` — amplitude (overall brightness)
- `x1` — stretch: how much of the first variation mode `M1` this SN shows
  (high x1 = broad, slow, intrinsically brighter; the Phillips relation)
- `c`  — color: scales one empirical color law `CL`
- `t0` — time of peak brightness

Four free parameters: `x0, x1, c, t0`. Fast to evaluate because `M0, M1, CL` are
precomputed surfaces you interpolate and multiply.

**Key weakness for lensing:** the single color parameter `c` mixes *intrinsic* SN
color and *host-galaxy dust reddening* into one number — SALT does not separate
them. For time delays this matters little; for magnification (and eventually H0)
it biases the brightness measurement.

Note on variants: use `salt2-extended` (covers ~3000–11000 Å rest-frame, spanning
all Roman IR bands) rather than plain `salt2` (cuts off ~9200 Å) or `salt3`
(narrower trained range).

### 1.2 BayeSN — the physically-motivated SN model

BayeSN warps a baseline SED (Hsiao) instead of adding principal components:

```
W    = W0 + theta * W1 + epsilon        # a warping surface in (phase, wave)
S    = S_Hsiao(phase, wave) * 10^(-0.4 * W)
flux = amplitude * 10^(-0.4 * M0) * S
```

Differences from SALT that matter here:

1. **theta** is BayeSN's analog of `x1` — the intrinsic shape/brightness parameter.
2. **Dust is a separate physical model, not a color term.** Reddening is applied as
   a real Fitzpatrick (1999) extinction law with free E(B–V) and free R_V (host)
   plus a fixed Milky-Way screen. This gives cleaner, less-biased magnitudes —
   exactly what you want for magnification and H0.
3. **epsilon** models intrinsic scatter; typically turned off for speed.

**Cost:** every flux evaluation does spline-interpolation matrix builds and a
Fitzpatrick extinction evaluation per dust screen — far heavier per call than
SALT's surface lookup. This is the per-evaluation expense behind the runtime.

### 1.3 SNTD — the fitting framework (shared by both)

SNTD is *not* a pipeline; it is the library that both pipelines call. It takes
*one* SED model (SALT or BayeSN) and fits it to *several images at once*, solving
for the time delays and magnification ratios between images. "SALT vs BayeSN" is
the wrong axis — the axis is *which SED model gets plugged into SNTD*.

SNTD offers three fitting strategies:

- **parallel** — fit each image *independently*, then get the delay as the
  difference of the independently-fitted peak times (t0_2 − t0_1), or by
  cross-correlating the two fitted models. Simple, robust, embarrassingly
  parallel. Throws away the "same SN" constraint. (Used by the SALT pipeline.)
- **series** — fit all images *jointly* with the delays and magnifications as
  *explicit shared parameters*, SN properties shared across images. More correct;
  gives the delay posterior directly. (BayeSN stage 1.)
- **color** — fit using band-to-band color information to pin the delay and cleanly
  separate dust from magnification. (BayeSN stage 2.)

### 1.4 The three nested layers of any fit

The recurring confusion — "when does it sample?" — is resolved by seeing that a
fit is three layers nested inside each other:

```
Layer 3  SAMPLER (dynesty / nested sampling, or emcee)   <- "the sampling"
   loops thousands of times, proposing whole parameter sets
      |
Layer 2  LIKELIHOOD (SNTD)
   for one trial parameter set, predict flux at every obs, score vs data -> one number
      |
Layer 1  SED MODEL (SALT3 or BayeSN _flux)
   given parameters, generate the whole light curve across all epochs/bands at once
```

Total cost ≈ (sampler steps) × (observations scored each step) × (one SED eval each).
That product is why it is minutes per system.

**Crucial correction to a common misreading:** the SED model does *not* fit
per-epoch. One parameter set produces the *entire* light curve across all epochs
and bands simultaneously; the sampler tries many *whole* parameter sets. What
varies between iterations is the *parameters*, not the epoch.

### 1.5 Where the two pipelines differ (and why BayeSN is slower)

| | SALT pipeline (`sntd_wrapper.py`) | BayeSN pipeline |
|---|---|---|
| Driver | `run_sntd_population.py` → `measure_one` | `run_bayesn_population.py` |
| SED model | SALT2-extended (in sncosmo) | `bayesncosmo.py` (`BAYESNSource`) |
| SNTD method | `parallel` (cheapest) | `series` then `color` (two most expensive) |
| Live points | ~100 | 500 each, run twice |
| Is Δt a sampled parameter? | **No** — derived from per-image t0 | **Yes** — `dt_a`, `dt_c` sampled directly |
| Dust/magnification | conflated in `c` | cleanly separated |
| Speed | fast, coarse | slow, careful |

The runtime gap is *larger* than the SED-cost difference alone: BayeSN also uses
heavier methods, 5× the live points, and two full runs.

### 1.6 The two-stage BayeSN fit is already a "pre-fit then tighten priors"

BayeSN's `series` → `color` handoff is itself the pattern the new design
generalizes: run 1 (`series`) samples theta + t0 jointly; its results become a
*Gaussian prior on theta* and *tight bounds on t0* for run 2 (`color`), which then
explores a much smaller space and converges faster. The new design pushes this
same idea one level earlier (to the delays) and to cheaper estimators.

---

## 2. The staged workflow

Four stages, each narrowing the search for the next. **That narrowing is where the
speed comes from.** The physical fit at the bottom is the simulation-independent
anchor that keeps the whole thing honest.

```
Stage 0 · slsim simulation
   thousands of systems with known delays, magnification, injected microlensing
   -> serves as BOTH transformer training labels AND validation ground truth
        |
Stage 1 · GP cross-correlation           ~seconds/system
   coarse Δt + flux ratio, no SN model
        |  (passes Δt hint + uncertainty)
Stage 2 · Transformer                     ~milliseconds/system
   refined Δt, magnification ratios, microlensing, SN params — each with error bar
        |  (passes tight bounds + priors)
Stage 3 · SNTD / BayeSN physical fit      ~minutes/system (but faster than today)
   sampler starts inside tight bounds -> publication-grade numbers
```

### Recommended build order (lowest risk first)

1. **Extract the flux ratio from the existing fit** (a few lines; see §7). Immediately
   useful, unblocks the magnification benchmark.
2. **Stage 1 GP module** — coarse Δt + flux ratio.
3. **Prime Stage 3 with it** and measure the speedup (upgrades `fast` from risky
   to trustworthy — see §5).
4. **Stage 0 simulator** with labeled microlensing injection.
5. **Gradient-boosted-tree baseline** on GP summary features — tells you what the
   transformer must beat.
6. **Transformer** last.

---

## 3. Stage 1 — GP cross-correlation (what it means)

The name is two techniques stitched together.

**Cross-correlation:** slide image B's curve past image A in time; at each trial
shift, measure how well they overlap. The shift with the best overlap is the
delay estimate. Generic signal processing — nothing SN-specific.

**Gaussian process (GP):** the two images aren't sampled at the same times (A at
day 10, B at day 12), so you can't compare them point-for-point. A GP draws a
*smooth curve with an error band* through scattered points, so you can ask "what
was B's flux at day 10?" and get an interpolated value *plus* an uncertainty. It's
a principled smooth interpolator that reports its own confidence in the gaps.

**GP cross-correlation** = GP-smooth each image into a continuous curve, then
cross-correlate the two smooth curves to find the aligning shift.

- No SN model, no sampler, no likelihood — just interpolate and slide.
- Outputs (per system, in seconds): **coarse Δt ± σ** and **flux ratio**
  (peak of A ÷ peak of B, a first proxy for magnification ratio).
- Caveat: gives Δt and flux ratio only; does **not** separate macro from micro or
  dust. It is a fast *estimator and prior generator*, not a replacement for the
  physical fit.

---

## 4. Stage 2 — the transformer

### 4.1 Why a transformer (and not a CNN/RNN/plain net)

A lensed SN system is an awkward ML input: irregular observation times, several
bands, multiple images, a *different number of observations per system*. A plain
net needs fixed-size input; a CNN wants a regular grid; an RNN struggles with
irregular spacing and interleaved images.

A **transformer consumes a variable-length *set* of tokens** — which matches "a
pile of photometric observations." Its **self-attention** compares every
observation with every other, which is *literally* how a time delay reveals
itself: image B's peak matches image A's peak, and the *time separation* of that
strong match **is the delay**. The architecture's core operation matches the
physics of the problem.

### 4.2 Input — turning light curves into tokens

The network reads a *set of rows*, one row (token) per photometric measurement:

```
token = [ phase, flux, flux_error, band_onehot, image_onehot, is_real_flag ]
```

Feature by feature, with the why:

- **phase** — time measured *relative to the GP-estimated peak* and divided by
  (1 + z_source) to get rest-frame phase. This makes every SN's intrinsic timescale
  look the same. (The *measured* delay you report stays in the observed frame —
  that's what carries the H0 information.)
- **flux** — normalized *per system* by that system's brightest detection, so the
  network sees light-curve *shape*, not absolute brightness.
- **flux_error** — fed in explicitly, so the network learns to down-weight noisy
  points itself.
- **band** — one-hot over the Roman bands. **Essential for microlensing** because
  microlensing is chromatic.
- **image_id** — one-hot over images. **Essential** because microlensing is
  per-image while the SN and dust are shared across images.
- **is_real_flag** — 1 for a real detection, 0 for padding (systems have different
  token counts; pad to a fixed length and mask the padding).

Per-system features appended to the pooled representation: `z_source`, `z_lens`,
GP coarse Δt, GP flux ratio, number of images.

> **Normalization warning.** Do *not* normalize flux to absolute magnitude via a
> cosmology — the distance modulus folds in the magnification, which is one of the
> things you are trying to measure. Normalizing that way silently divides out the
> signal. Use a per-system internal reference (brightest detection) instead, and
> keep redshift as a separate input feature.

### 4.3 What the network does internally

1. **Embed** each token into the working dimension.
2. **Self-attention** stack: every token compares to every other. This is where
   "image B = image A shifted by Δt" and "this color-drift differs between images"
   get discovered.
3. **Pool** to a single system vector (a learned summary token).
4. **Fuse** with the per-system scalar features (redshifts, GP hints).
5. **Heads** predict each quantity.

### 4.4 How it separates macro from micro lensing

This is the scientifically novel capability. It works because the two effects
leave *different, learnable fingerprints*, and because multi-image structure lets
the network isolate them:

- **Macro-lensing** (magnification): achromatic (same factor at all wavelengths),
  constant in time — a single flat brightness factor.
- **Micro-lensing:** chromatic (bluer wavelengths bent more, because the SN's
  blue-emitting region is physically smaller), and time-varying (drifts as the
  caustic geometry changes).

The give-away: the **SN itself and host-galaxy dust are shared by both images**
(same object, same host), so they produce *identical* signals in A and B.
**Microlensing is per-image** — each image travels a different sightline through
different lens-galaxy stars. So the recipe the network learns is:

```
align images in time (using the delay)
subtract everything shared by both images (the SN, the dust)
remove the flat achromatic macro factor
=> whatever color-dependent, per-image wiggle remains is the microlensing
```

Pre-aligning with the GP Δt makes this residual pop out. The network can only
learn this because training data contains *labeled* examples with known injected
microlensing — which is exactly what the staged-injection simulation provides.

### 4.5 Output — multiple heads, each with uncertainty

Each head outputs a value *and* a variance (heteroscedastic / aleatoric
regression, trained with Gaussian negative-log-likelihood), so every prediction
carries a calibrated error bar.

| Head | Output | Role |
|---|---|---|
| Time delay | Δt per image pair + σ | primary measurement; becomes Stage 3 bounds |
| Magnification ratio | μ_i / μ_1 per image + σ | macro-lensing; feeds H0 work |
| Microlensing | per-image amplitude + chromatic slope + σ | the novel capability; flagging-grade until validated |
| SN parameters | theta/x1-like, dust/A_V-like + σ | become Stage 3 priors |

Predicting these *jointly from one shared representation* lets the network use its
understanding of one to constrain the others (microlensing masquerades as color,
color as dust) — which the sequential physical fit cannot do cleanly.

### 4.6 Training

Supervised regression against slsim truth. Loss = sum of per-head Gaussian NLL,
with a per-target mask so absent effects (e.g. no microlensing in a given tier)
don't create spurious gradients.

The load-bearing validation is not the held-out test set — it is:
- **Calibration:** when a head says σ = 2 days, is it right ~68% of the time
  within 2 days? An overconfident microlensing flag is worse than none.
- **Cross-check against Stage 3** on real-data-like systems.

---

## 5. Stage 3 — the physical fit, and the fast/robust modes

`fast` and `robust` are **both Stage 3** — both the physical fit (SNTD + SALT2),
differing *only* in how wide a t0 search window they hand the sampler. Neither is
the GP; the GP feeds them.

- **robust** — t0 bounds span every image's own significant-SNR detection window,
  padded; full nested sampling, no cap. Safe against a bad initial guess, slow.
- **fast** — small t0 window re-centered per image by a quick `sncosmo.fit_lc`
  pre-fit; npoints/maxcall capped. Fast, but can lock onto the wrong peak
  (aliasing, low S/N) and then the narrow window prevents recovery.

**How the GP upgrades `fast` from risky to trustworthy.** The reason `fast` is
risky today is that its self-recentering (`sncosmo.fit_lc`) fits *each image alone*
and can center the narrow window on the wrong peak. The GP's cross-correlation
instead looks at the *relationship between the two images* directly, so it places
the window reliably. With a GP Δt in hand:

- The narrow `fast` window is centered on a *trustworthy* delay estimate — no
  wrong-peak risk.
- The GP's uncertainty tells you *how wide* to make the window per-system, instead
  of a fixed `fast_t0_window`.

**Decision rule:**
- Today, without the GP: use **robust** (fast's self-recentering can fail silently).
- With the GP: use **fast**, primed by the GP Δt, as the default. Keep **robust**
  as the fallback for systems where the GP cross-correlation peak is broad or
  double-peaked — which is itself a useful "this one is hard, spend more time"
  signal.

Concrete handoff: GP outputs Δt ≈ 47 ± 5 d → set `fast`'s t0 window to cover that
range with margin → SNTD's sampler starts in the right place → converges fast,
no wrong-peak risk.

---

## 6. Stage 0 — the simulator and the staged-injection idea

Everything ML depends on the simulator, so this stage is load-bearing.

Generate 10k+ systems with *known* truth: delays, macro-magnification, injected
microlensing curves, SN parameters, Roman cadence + noise. Deliberately *vary*
the things you worry about (microlensing strength, cadence gaps, host
contamination, redshift) so the network sees the full range and validation tests
robustness rather than only easy cases.

**Staged injection (ablation tiers).** Generate matched systems that differ only by
which effects are present:

```
SN only
SN + macro
SN + macro + micro
SN + macro + micro + milli
```

The *right* way to use these (not four separate networks):

- **One network**, trained on data spanning all tiers, **each example labeled with
  the true value of each effect** (zero when absent).
- **Curriculum learning:** start on the easy tiers (learn the clean delay first),
  phase in micro/milli. Networks often learn a hard disentangling task better when
  eased in.
- **Ablation diagnostics:** matched pairs differing only by microlensing let you
  measure exactly how much microlensing degrades the delay and whether the micro
  head recovers it — a controlled experiment built into the training set, and the
  evidence a referee will demand. (The physical fits *cannot* be trained on such
  matched pairs.)

Treat the **millilensing** tier (lens substructure / dark-matter subhalos) as the
most speculative label — hardest to simulate realistically and to validate. Keep
it as research-grade output, below microlensing in how much you trust it.

---

## 7. Immediate fix — extract the flux ratio you already compute

The current SALT pipeline outputs *only* the time delays. The magnification
information exists inside the fit — SNTD fits each image's `x0` (amplitude), and
the *ratio of the two x0 values is the magnification ratio* — but it is currently
computed and thrown away.

After the fit, per image, pull the fitted `x0` and take the ratio:

```
# magnification ratio image_i / image_1 = x0_i / x0_1
# (exact accessor depends on SNTD version — the fitted per-image model's x0)
```

Log it alongside the delays. **Caveat:** this `x0` ratio is the *total*
magnification (macro × micro). With no microlensing in the current sim it equals
the macro ratio — but only because micro is absent, not because the fit separates
them. Separating them is exactly what Stage 2 is for.

---

## 8. The honest caveats (so nothing blindsides you)

- **Simulation realism is the whole ballgame.** The network only knows what slsim
  taught it. If real microlensing or SN diversity differs from the simulator, the
  network can be confidently wrong in ways a physical fit would not be. This is the
  central scientific risk.
- **Microlensing output is flagging-grade** until calibrated against Stage 3 and
  real systems. Do not quote it as a measurement early.
- **Build the boring baseline first** (GP + gradient-boosted tree on summary
  features). If it meets your Δt accuracy bar, the transformer may be unnecessary
  for delays; if not, it tells you exactly what the transformer must beat.
- **The transformer is the accelerant, not the foundation.** Stages 0, 1, and 3
  give a working, publishable, faster pipeline with *no ML at all*. That is the
  right risk ordering.
- **Defensible framing for a paper:** robust mode is the ground-truth-independent
  anchor; fast mode (GP-primed) is the throughput workhorse; the residual
  distribution between them is itself a publishable result quantifying the accuracy
  cost of speed.

---

## 9. Two "parallel" words that are easy to confuse

Three independent knobs share overlapping names — keep them straight:

- `N_JOBS` (joblib, in the driver) — how many *systems* run at once across cores.
- `method="parallel"` (SNTD) — a *statistics* choice: fit images *independently*
  vs jointly. No cores involved.
- `ncpu` (dynesty, inside one fit) — how many cores one fit's sampler uses
  internally.

BayeSN currently runs one system at a time but throws many cores at that one
system's sampler; the SALT driver (with `N_JOBS=-1`) runs many systems at once,
each fit modest. Different parallelization philosophies.
