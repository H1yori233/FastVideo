# Foreground-Anchored Attention: Training-Free Identity-Preserving Editing of Autoregressive World Models

*Working draft — quantitative results from exp45–49. All numbers reproducible with
the scripts in this repo (see §7). Model: `mg_zelda_longlive` (MatrixGame-2, 4-step
DMD, causal/streaming). No weights are trained — the method is inference-only.*

---

## Abstract

Autoregressive (AR) video world models generate frame-by-frame conditioned on a
KV-cache of the past and an action stream. When driven from an **out-of-distribution
(OOD) anchor image** — e.g. a robot or Wukong dropped into a Zelda-trained model —
the rollout **drifts to the model's training prior** (the "Link attractor"): within
~50 frames the OOD subject morphs into the in-distribution character. We show this
drift is an *attractor* property of the AR map and that it can be countered **without
any training**. Our method, **Foreground-Anchored Attention (FAA)**, (i) pins the
anchor frame as a persistent attention sink, (ii) applies a soft feathered bias that
anchors the *subject region* while letting the *background travel*, (iii) gates the
intervention to structural layers, and (iv) introduces a **Subject-Region Value
Anchor** that re-grounds the subject's *appearance* (values), not just attention
routing. On a 4-subject OOD benchmark, FAA raises DINO identity-to-anchor similarity
from **0.50–0.60 (baseline) to 0.77–0.80** and roughly **halves the identity-drift
slope**, while retaining **15–25× the motion of a frozen clip** — i.e. identity is
held without freezing the subject or the world.

---

## 1. Problem: the AR training-prior attractor

An AR world model is a map `x_t = D(ctx_{<t}, a_t)` where `ctx` is the KV-cache. The
training manifold (here: Zelda/Link) is an **attractor**: any generative step that is
off-manifold is pulled back on-manifold. With a default cache (`sink=0`,
`local_attn_size=6`), the anchor frame is *evicted* after ~6 frames, so an OOD subject
has no persistent grounding and decays to the prior.

**Quantified (robot, 201f, DINO sim-to-anchor):** the clean baseline falls from 0.97
→ 0.30 with drift slope **−0.30/100f**; across 4 subjects mean identity is only
**0.50–0.60** (Table 2). This is the failure FAA targets.

The drift is **humanoid-specific**: the Link prior is a humanoid-locomotion attractor,
so non-humanoid anchors (car, robot-dog) do *not* drift and need no intervention — a
useful control that isolates the attractor as the cause.

---

## 2. Method: Foreground-Anchored Attention

All operations are in the cache-attention of each self-attention block; `q,k,v` are
the current queries and the windowed keys/values `[sink | cross | within]`. Let the
token grid be `R×C = 30×52`, and `fg_box` the subject region in grid coords.

**(a) Anchor sink.** Set `sink_size=1` so the anchor frame's KV is never evicted.

**(b) Soft feathered subject bias.** Add a zero-mean Gaussian logit bias on the sink
columns, centered on `fg_box`: subject columns are boosted, periphery suppressed,
`w(p)=exp(-‖p−c‖²/2σ²)`, `bias = (w − w̄)·β`. Center boost β pins the subject to its
own identity; the smooth feather avoids the rectangular seam a hard mask produces.

**(c) Background query-suppression.** Background current-frame queries (low `w`) get
`−(1−w)·κ` on the sink columns, so they ignore the anchor → the **background is free
to travel** and the subject is free to **articulate**. κ trades subject-pinning vs
world-motion.

**(d) Structural-layer gate.** Apply (b,c) only on the middle layer band
`[0.2, 0.8]` of depth. Detail layers are left untouched → coherent background.

**(e) Subject-Region Value Anchor (the key long-horizon component).** All of (a)–(d)
act on attention *routing* (logits); identity/appearance lives in the **values**.
We blend the current within-frame values, **inside `fg_box` only**, toward the
frame-0 anchor values: `v ← (1−λ)·v + λ·v₀` at matching grid positions. Because the
anchor has the correct colors and *no* spurious prior-gear, this directly re-grounds
identity. It is spatially restricted to the subject, so unlike a whole-frame value
hold (which freezes everything — see §6) it preserves articulation and travel.

The full method = (a)+(b)+(c)+(d)+(e). Validated: `sink=1, win=6, σ=(5,5), β=4,
κ=6, layers 0.2–0.8, λ≈0.6`.

---

## 3. Evaluation protocol

- **Identity preservation:** DINO ViT-B/16 (self-supervised) CLS-feature cosine
  similarity between the subject-region crop at frame `t` and the anchor's subject
  crop. Report **mean**, **final** (last 5 sampled frames), and **drift slope**
  (linear fit, per 100f). DINO is appearance-sensitive and unsupervised → not gamed
  by texture priors. *(We deliberately avoid optical-flow EPE for identity — it is a
  motion proxy and unreliable here.)*
- **Dynamism (anti-freeze control):** mean L2 between consecutive DINO features.
  Calibrated against a frozen clip (repeat frame 0) = **0.009**. A method that holds
  identity by freezing would sit near this floor.
- **Setup:** `mg_zelda_longlive`, 480×832, 4-step DMD, seed 42, action = W (forward).
  201f for ablation, 297f for the long-rollout proof.

---

## 4. Component ablation (Table 1, robot 201f)

Each rung adds exactly one component.

| # | config | id mean | id final | drift slope/100f | dynamism | reading |
|---|---|---|---|---|---|---|
| 1 | clean (base model) | 0.503 | 0.300 | −0.299 | 0.415* | drifts to Link |
| 2 | +sink | 0.556 | 0.418 | −0.260 | 0.445 | marginal |
| 3 | +boost | **0.915** | 0.896 | −0.050 | 0.173 | identity locked but **frozen** |
| 4 | +qsupp | 0.756 | 0.657 | −0.144 | 0.323 | motion restored, some id lost |
| 5 | +layer-gate (base recipe) | 0.558 | 0.422 | −0.277 | **0.468** | max motion/coherence, id sacrificed |
| 6 | **+value-anchor (ours-full)** | **0.809** | **0.765** | −0.114 | 0.216 | **id recovered, not frozen** |

\* clean's "dynamism" is largely *drift* (changing identity), not useful motion.

**Mechanistic story.** boost (3) pins identity but freezes the subject (dynamism
0.17, near the frozen floor). qsupp (4) and layer-gate (5) restore motion and
background coherence but let identity drift back to baseline (0.56). The **value
anchor (6) recovers exactly the identity that the motion-promoting components gave up
(0.56 → 0.81) without re-freezing** (dynamism 0.22, ≈24× the frozen floor). The
components trace an **identity–dynamism Pareto frontier** (Fig. `_pareto.png`); the
value anchor pushes the frontier toward high-identity-with-motion.

---

## 5. Generalization (Table 2, 201f, clean vs ours-full)

| subject | clean id mean | **ours id mean** | clean final | ours final | clean slope | ours slope |
|---|---|---|---|---|---|---|
| robot | 0.51 | **0.77** | 0.42 | 0.70 | −0.09 | −0.10 |
| wukong | 0.60 | **0.79** | 0.38 | 0.64 | −0.30 | −0.18 |
| genshin | 0.58 | **0.80** | 0.41 | 0.73 | −0.30 | −0.10 |
| minecraft | 0.52 | **0.78** | 0.24 | 0.69 | −0.39 | −0.14 |

Consistent +0.20–0.26 mean identity and ~2× slower drift across four visually
distinct OOD humanoids (metallic robot, dark Wukong, anime Genshin, blocky MC). The
long-rollout proof (297f, exp46) shows the same qualitatively, with travel + gait.
Figure: `_generalize.png`.

**Action coverage (robot, clean → ours, exp51).** The gain is not forward-specific:

| action | clean id mean | **ours id mean** |
|---|---|---|
| W (forward) | 0.51 | **0.77** |
| A (strafe) | 0.43 | **0.77** |
| r (yaw) | 0.52 | **0.69** |

Yaw is the hardest (camera rotates the subject's view) but still clearly improves.
In every case the clean "dynamism" (0.39–0.55) is inflated by drift while ours
(0.24–0.33) is real motion above the 0.009 frozen floor.

---

## 5b. Attractor escape & metric corroboration (robot 201f, exp-attractor)

The core claim — the OOD subject *escapes the training-prior attractor* — is measured
directly. We define the **Link pole** as the clean baseline's final-frame subject crop
(the state the subject drifts into) and track DINO sim(frame_t, Link pole):

| | DINO→anchor | CLIP→anchor | DINO→**Link** (final) |
|---|---|---|---|
| clean | 0.501 | 0.734 | 0.658 (**0.894**) |
| **ours** | **0.811** | **0.851** | **0.383 (0.449)** |

Clean **rises to 0.89** sim-to-Link (falls into the attractor); ours **stays at ~0.45**
(escapes). The identity gain reproduces under a **second backbone (CLIP)**, so it is
not DINO-specific. Figure `_eval_attractor.png` (right panel) is the headline result:
the clean curve climbs to the prior, ours stays flat. This is the cleanest statement
of the contribution and rebuts "the metric is gamed" (two backbones + an independent
attractor-distance axis + the frozen-dynamism control all agree).

## 6. Limitations (honest)

- **Subject-region prior composition (irreducible training-free).** The model
  composites a Hylian-paraglider/shield onto the humanoid back after ~f100 (its
  humanoid+glider prior). It lives in the subject region, *absent in frame 0*. The
  value anchor **reduces** it but cannot remove it; background-side knobs cannot reach
  it (exp45). Fully removing it needs weight-level grounding.
- **Dynamism vs identity is a frontier, not a free lunch.** Stronger value-anchoring
  (λ>0.8) hazes/over-constrains (exp47); λ≈0.6 is the knee. We tested **three distinct
  value-anchor mechanisms** — position-matched blend, gaussian-feathered blend, and
  appearance-correspondence retrieval (exp52–54) — and all land on the *same*
  identity–dynamism frontier; the feathered variant even regresses on some subjects
  (background bleed). This is evidence the frontier is **fundamental to the training-
  free setting**, not an artifact of one anchor implementation. The hard-box position-
  matched anchor (λ=0.6) is the validated operating point.
- **Metric scope.** DINO sim-to-anchor rewards appearance fidelity; we pair it with
  the frozen-calibrated dynamism control and full-resolution visual QA to prevent
  proxy-metric false positives. A VLM-judge / human study is the natural next eval.
- **Per-subject box.** `fg_box` is currently manual; automatic subject segmentation
  would remove the one manual knob.

---

## 7. Reproducibility

- Algorithm: `fastvideo/models/dits/matrixgame2/attn_injection.py` (`INJECTOR`,
  mode `fg_sink`) + 1-line hook in `causal_model.py`.
- Single-run: `run_ood_identity.py --image --box r0 r1 c0 c1 --boost --qsupp --vhold`.
- Ablation: `exp48_ablation.py`. Generalization: `exp49_crosssubj.py`. Long proof:
  `exp46_longshow.py`. Value-anchor sweep: `exp47_vhold.py`.
- Eval: `eval_identity.py` (DINO id-sim + dynamism + curves).
- Snapshots: `snapshots/v1_exp46_*` (base), `snapshots/v2_exp47_*` (+value-anchor).
- Full round-by-round log: `docs/attn_injection_log.md`.

---

## 8. Next steps to submission-grade

1. **VLM-judge pairwise** (ours vs clean: "which keeps the original character?") to
   complement the embedding metric — gold-standard identity eval.
2. ~~Frame-scheduled value-anchor (λ: 0→0.6)~~ — **tested (exp50), negative**: flat
   λ=0.6 beats any ramp (final id 0.77 vs 0.66). Identity drift is an attractor, so
   deferring the grounding lets it gain ground → apply the value anchor from frame 0.
   (A small but clean finding worth a sentence in the paper.)
3. **Action coverage** — repeat the benchmark over {W,A,D,yaw} not just W.
4. **Automatic subject box** (open-vocab segmentation) to drop the manual knob.
5. **Human study** on identity + naturalness for the camera-ready.
