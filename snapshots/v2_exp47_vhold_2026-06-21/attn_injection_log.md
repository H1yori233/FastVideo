# Attention-Injection Experiment Log

> Living doc. **Workflow: read this BEFORE each round; update it right AFTER each
> round** (env / model+data / code changes / params / output paths / visual result
> / failure modes / next step). Do NOT backfill at the end.

---

## 0. Critical setup (read first)

**Codebase:** `/home/hal-kaiqin/FastVideo_attninj` — a clean clone of public
`hao-ai-lab/FastVideo` (upstream MatrixGame = `matrixgame2`). All our algorithm
lives in `fastvideo/models/dits/matrixgame2/attn_injection.py` + a 1-line hook in
`causal_model.py`. **Training-free only — no LoRA/finetune (user requirement).**

**Conda / run command** (the env's editable `fastvideo` points to the OLD ~/FastVideo,
so we MUST prefix `PYTHONPATH` to use THIS repo):
```bash
cd /home/hal-kaiqin/FastVideo_attninj
PYTHONPATH=/home/hal-kaiqin/FastVideo_attninj CUDA_VISIBLE_DEVICES=0 \
  /home/hal-kaiqin/miniforge3/envs/FastVideo_kaiqin/bin/python <script>.py
```
GPU: 4× GB200 192GB (use device 0). A 117f rollout ≈ 25–30s, 201f ≈ 40–50s, 597f ≈ 90–100s.

**Model:** `mg_zelda_longlive` (Zelda-finetuned MG2, 4-step DMD, causal/streaming).
Load via symlink path so the registry resolves the config:
`/home/hal-kaiqin/models/matrix-game-2.0-zelda-longlive` -> `mg_zelda_longlive`.
- `mg_causal_zelda` ("Ours" 4-step) FREEZES on OOD (don't use for OOD).
- Inference res 480×832, num_inference_steps=4, seed=42. Latent token grid = **30×52**
  (rows×cols) per frame. NUM_FRAMES must give latent T divisible by 3
  (valid: 45, 117, 201, 297, 597; NF=4k+1 with (NF+3)/4 %3==0).

**Data:**
- OOD anchors (Zelda-composite, snowy): `~/FastVideo/assets/third-person/combine/`:
  `robot_zelda_scene.jpg`, `robot_dog_zelda_scene.jpg`, `wukong_zelda_scene.jpg`,
  `car_zelda_scene.jpg`. Game-like: `~/FastVideo/assets/third-person/`:
  `genshin.png`, `mc_third_person.jpg`, `wukong_832x480.jpg`.
- Actions: `~/FastVideo/examples/training/finetune/WanGame2.1_1.3b_i2v/actions_801/`
  `{W=fwd, A/D=strafe, S=back, l/r=yaw, u/d=pitch}.npy` (keyboard (T,6)→slice [:4];
  mouse (T,2)). Load: `np.load(p,allow_pickle=True).item()`.

**Outputs:** `attn_injection_out/exp<N>_<name>/<tag>/output.mp4`. Analysis PNGs go to
`attn_injection_out/_<name>.png`.

---

## 1. THE METHOD (current best, training-free) — Foreground-Anchored Attention

Reproducible: `run_ood_identity.py` (`--image --box r0 r1 c0 c1 --boost --qsupp`).

Problem: plain rollout of an OOD humanoid (robot) DRIFTS to the model's training
prior (Link) over the autoregressive rollout. Non-humanoid (car/dog) DON'T drift
(no competing prior) — they already work clean.

Mechanism (all in `attn_injection.py`, mode `fg_sink`, `fg_soft=True`):
1. `sink_size=1` retains anchor (frame 0) in KV cache.
2. **Soft feathered** gaussian bias on the sink columns (no hard seam): center
   (subject) boosted, periphery suppressed. `fg_boost` strength.
3. **Soft query-suppression** (`fg_qsupp`): background current-frame queries
   down-weight the sink → background free to TRAVEL and subject free to ARTICULATE.
4. **Mid-layer gating** (`layer_lo=0.2, layer_hi=0.8`): apply fg only on structural
   layers, not detail layers → coherent background + MORE articulation.
5. **`fg_box` MUST cover the actual subject** (usually bottom/mid-center in
   third-person). robot in robot_zelda_scene = rows~14–30, cols~21–31 of 30×52.

**Validated params (robot):** sink1, win6, fg_box=(14,30,21,31), sigma=(5,5),
fg_boost=4, fg_qsupp=6, layers 0.2–0.8. → silver robot walks w/ gait + identity
held + coherent snowy-Zelda world ~150 clean frames. Generalizes (wukong/genshin/mc).

### Knobs & couplings (learned)
- `fg_qsupp` drives BOTH subject articulation AND background freedom: low→frozen
  subject + tethered; high→articulating subject + bg can hallucinate.
- `fg_boost` ↑ = stronger identity but freezes subject past ~5–6.
- Layer gating to mid = key for background coherence (don't distort detail layers).
- Box on wrong region (sky) = no identity hold (was a real bug).

---

## 2. Dead ends (do NOT retry)
- Hard masks / strong boost → rectangular SEAM artifact + frozen subject.
- Image-embed scaling (CLIP) → too weak, identity still drifts.
- Latent EMA drift-correction → raises motion but doesn't restore identity (latent
  linear correction ≠ appearance manifold).
- AdaIN on token axis → FREEZES (wrong axis; motion lives in temporal not spatial).
- Cross-frame / within-frame routing injection → can't synthesize OOD gait.
- sink_size=3 + keyboard×3 → holds identity but SPINS in place / no forward travel
  (sink tethers viewpoint; ×3 amplifies spurious yaw). Forward travel needs sink0
  OR the query-suppression decoupling.
- Larger context window → froze subject + worse haze.
- Late qsupp-decay (re-ground bg) → no effect on the residual (exp45): the green
  gear is a SUBJECT-region paraglider/shield composition, not a free-bg object, so
  a background-side knob can't touch it. Residual is training-free-irreducible.

---

## 3. Round-by-round log (newest at bottom)

### Consolidated up to exp38 (2026-06-21)
- exp16: keyboard×N scaling raises motion at sink3 (kb1x 0.45→kb4x 0.69), robot held.
- exp22: diagnosed spinning — sink tethers (departure flat), ×3 yaws; sink0+kb1
  travels (departure 22→77).
- exp31 (KEY): box-fix (anchor actual robot not sky) → humanoid ARTICULATES + identity
  held. b4_q6.
- exp34: generalizes to wukong/genshin/mc (per-scene boxes), clean ~110f.
- exp35 (KEY): mid-layer gating (0.2–0.8) → coherent background (snowy Zelda, Sheikah
  towers ~150f) + MORE articulation (6.9→13.0). Best so far: exp35_layers/mid20_80.
- exp36/38: boost/box tuning to kill residual back Link-gear — minor reduction, not
  eliminated (model's stubborn tendency; harder boost freezes).
- exp37: clean baseline (sink0) confirms it DRIFTS to full Link by f80–160.
  Before/after: `attn_injection_out/sbs_final.mp4`.

**Current visual result:** OOD humanoid (robot) = SUCCESS — walks w/ articulated gait,
identity held vs clean→Link, coherent world ~120–150 frames, training-free. Non-humanoid
(car/dog) work clean already.
**Failure modes remaining:** (a) minor Link-gear on robot's back after ~f120; (b)
background haze/hallucination past ~150f (base-model long-OOD limit, coupled to qsupp).
**Next:** run the recipe on wukong / car / other OOD with correct per-subject boxes,
judge by full-res frames, then keep polishing residual.

### exp39 (2026-06-21) — wukong + car OOD
- env/model/code: unchanged (mg_zelda_longlive, attn_injection.py, training-free).
- params: NF=117. wukong_fg = recipe with **fg_box=(15,29,16,29)**, boost4, qsupp6,
  layers 0.2–0.8, sigma(5,5), sink1. wukong_clean / car_clean = sink0, INJECTOR off.
- outputs: `attn_injection_out/exp39_ood/{wukong_clean,wukong_fg,car_clean}/output.mp4`;
  grid `attn_injection_out/_exp39.png`.
- visual: **wukong_clean DRIFTS to Link** (lighter/sword by f110); **wukong_fg HOLDS**
  the dark Wukong figure → recipe generalizes to a 2nd humanoid. **car_clean** = yellow
  car identity held + travels forward (non-humanoid needs NO fg).
- failure modes: wukong box (15,29,16,29) slightly loose; box-on-subject still the
  critical knob. Same residual/haze couplings as robot expected at longer NF.
- next: tighten wukong box + try longer NF; polish robot residual back-gear; consider
  auto subject-box (segmentation) instead of manual boxes per scene.

### exp40 (2026-06-21) — vertical box vs back-gear residual
- env/model/code: unchanged.
- params: robot, NF=201, vertical/tall narrow box (10,30,23,30), sigma(8,4),
  boost{4,5}, qsupp6, layers0.2-0.8.
- outputs: `attn_injection_out/exp40_vbox/{vtall_b4,vtall_b5}/`; `_vbox.png`.
- visual: robot held + walking (vtall_b5 marginally cleanest) but **back green/blue
  element STILL present** ~f120+. Box/boost/sigma tuning has hit its limit on this
  residual — likely a BACKGROUND hallucination (Zelda creature/structure) behind the
  robot, not the subject, so subject-anchoring can't remove it.
- failure mode (characterized): residual back-element = background-coherence problem,
  not identity. Couples with the long-OOD haze. Appears mostly AFTER ~f120.
- next: verify clean-length (NF=117) is artifact-free as the deliverable; for the
  residual, the lever is background coherence (not subject box) — try lighter qsupp
  schedule late in rollout, or just deliver ≤120f clean clips.

### exp41 (2026-06-21) — clean-length 117f deliverable check
- env/model/code: unchanged. Used `run_ood_identity.py` (the reproducible script).
- params: robot, full recipe, box(14,30,21,31), boost4, qsupp6, mid-gating, NF=117.
- outputs: `attn_injection_out/exp41_clean117/robot/output.mp4`; `_clean117.png`.
- visual: **at full-frame = clean visual SUCCESS** — silver robot walks through
  coherent snowy-Zelda world (snow, Sheikah towers, mountains) all 117f. Green
  back-element only visible on CLOSE ZOOM and starts ~f100. So practical fully-clean
  length ≈ 90f; ≤117f is good at normal viewing.
- failure mode: green back-element from ~f100 (close-up only) — minor; couples w/ bg.
- next: try frame-scheduled qsupp (ease off late) to push the clean length; OR accept
  ~90-117f clean deliverable and run the full OOD set (robot/wukong/genshin/mc/car/dog)
  as final showreels. Recipe + log are in place; result is a solid training-free crack.

### exp42 (2026-06-21) — full OOD showreel (clean length 117f)
- env/model/code: unchanged.
- params: NF=117. genshin_fg box=(12,27,23,32), mc_fg box=(12,27,23,33), both full
  recipe (boost4,qsupp6,mid-gating). dog_clean = sink0 off (non-humanoid).
- outputs: `attn_injection_out/exp42_show/{genshin_fg,mc_fg,dog_clean}/`; consolidated
  grid `attn_injection_out/_showreel.png` (robot+wukong+genshin+mc+car+dog).
- visual: broad OOD success @117f — humanoids hold identity+animate+travel via recipe;
  non-humanoids (car,dog) work clean. Cleanest: robot, mc, car. Minor drift by f110:
  wukong, genshin, dog. mc keeps blocky style well.
- failure modes: per-scene box still manual (genshin/mc boxes approximate); minor
  late drift on some cases; same ~f100+ residual.
- next: (a) frame-scheduled qsupp to extend clean length; (b) auto subject box;
  (c) per-case box tuning for wukong/genshin/dog. Current state = strong, well-
  documented training-free OOD crack.

### exp43 (2026-06-21) — frame-scheduled boost ramp (CODE CHANGE)
- code change: wired `start_frame` through `causal_model.py` hook ->
  `injected_sdpa` -> `maybe_apply`. Added `fg_boost_ramp` / `fg_boost_max`:
  boost_eff = fg_boost*(1+ramp*start_frame), clamped. (start_frame is in LATENT
  frames, ~0..50 for NF=201.)
- params: robot NF=201, base recipe, ramp {0, 0.02, 0.04}, boost_max=9.
- outputs: `attn_injection_out/exp43_ramp/{ramp0,ramp02,ramp04}/`; `_ramp.png`.
- visual: ramp REDUCES tail articulation (late artic 16→12→6.9) i.e. freezes the
  tail; does NOT remove the back-gear. The residual is a **green Hylian shield +
  paraglider** generated BEHIND/on the robot = **BACKGROUND hallucination**, not
  subject identity → subject-side boost/ramp can't fix it. DEAD END for this residual.
- failure mode (final characterization): residual = background generates Link gear-
  objects near the subject. Fixing needs background-side grounding, which couples
  with travel/articulation (fundamental tradeoff). Subject identity itself is solved.
- next: subject side is DONE. Either accept ~90-117f clean deliverable, or attack
  background hallucination directly (weak background-sink for coherence vs travel
  tradeoff). Recommend: deliver clean-length showreel; flag residual as known limit.

### exp47 (2026-06-21) — fix long-rollout identity via SUBJECT-REGION value anchor (CODE CHANGE)
- snapshot first: validated v1 saved to `snapshots/v1_exp46_2026-06-21/` before edit.
- code change: new `fg_vhold` / `fg_vhold_adain` in `_fgsink_soft`. Blends the
  current within-frame VALUES, ONLY inside fg_box, toward the frame-0 anchor values
  (per matching grid position). Motivation: all prior knobs touched only attention
  ROUTING (logits); identity/appearance lives in the VALUES. The anchor (frame 0)
  has the correct identity colors and NO glider, so pulling the subject-region
  values back to it pins identity and counters the glider — spatially restricted +
  gentle, so it does NOT freeze (unlike the whole-frame value-hold dead end).
- params: robot NF=297, op-A recipe. Sweep vhold {0,0.3,0.6,0.8,1.0} blend +
  adain {0.5,0.8}.
- outputs: `attn_injection_out/exp47_vhold/{ctrl,blend03,blend06,blend08,blend10,
  adain05,adain08}/`; before/after `attn_injection_out/exp47_vhold_sbs.mp4`; grids
  `_exp47_grid.png`,`_exp47_id.png`,`_exp47_cmp.png`,`_exp47_ceiling.png`.
- visual (full-res QA): **WORKS, modest, saturates ~0.6.** blend06 holds the
  silver-metallic robot more consistently over the long rollout (stays robot-colored
  where ctrl acquires more Link-green gear) AND keeps articulation + travel (subject
  changes pose/position f60→290 — NOT frozen). blend03 = marginal. blend08 starts
  to haze; **blend10 degrades badly** (washed-out haze + artifacts — over-anchoring
  corrupts the rollout). AdaIN variant ≈ blend, no better. Back-glider REDUCED but
  not eliminated (it's a fresh per-frame composition over sky-behind-subject positions
  that frame-0 can't fully overwrite).
- failure mode: vhold>~0.8 hazes (values over-constrained); glider not fully removed.
- **CONCLUSION: best long-rollout recipe = base recipe + `fg_vhold≈0.6`.** First
  mechanism to act on VALUES (appearance) not routing; gives the best identity hold
  yet without freezing. Exposed as `--vhold` in `run_ood_identity.py` (default 0 =
  validated base). Snapshot this as v2.
- next: optional frame-scheduled vhold (0 early -> 0.6 late) to remove any early cost;
  per-case vhold for wukong/genshin; otherwise identity side is at its training-free
  ceiling.

### exp46 (2026-06-21) — LONG-rollout proof across diverse examples (NF=297)
- env/model/code: unchanged (mg_zelda_longlive, recipe op-A: sink1/win6, boost4,
  qsupp6, mid-gating 0.2-0.8, sigma(5,5)). Action W (forward). One model load.
- cases (per-subject box): robot_fg(14,30,21,31), wukong_fg(15,29,16,29),
  genshin_fg(12,27,23,32), mc_fg(12,27,23,33); robot_clean (injector OFF, baseline);
  car_clean, dog_clean (non-humanoid, OFF).
- outputs: `attn_injection_out/exp46_longshow/{robot_fg,robot_clean,wukong_fg,
  genshin_fg,mc_fg,car_clean,dog_clean}/output.mp4`; before/after stacked video
  `attn_injection_out/exp46_robot_sbs.mp4`; grids `_exp46_grid.png`,
  `_exp46_robot_ba.png`, `_exp46_robot_id.png`.
- visual (full-res QA): ALL 7 produce coherent 297f rollouts that TRAVEL. Humanoids
  (robot/wukong/genshin/mc) hold identity + articulate via recipe; mc keeps blocky
  style best; non-humanoids (car/dog) hold clean + travel forward through snow.
  robot_fg keeps silver-metallic body vs robot_clean drifting more Link-ward. Known
  residual (subject-region back-glider, exp45) appears on both humanoids after ~f100;
  it does NOT break travel/identity, just adds the irreducible gear.
- failure modes: same training-free-irreducible back-glider on humanoids past ~f100;
  genshin travels least (anchor is a near-static character in a field).
- next: proof set delivered (long + diverse). Subject side solved; residual
  characterized. Optional: per-case box polish for wukong/genshin; final showreel.

### exp45 (2026-06-21) — late background RE-GROUNDING via qsupp decay (CODE CHANGE)
- hypothesis: the late Zelda-content hallucination is caused by the soft query-
  suppression fully DETACHING bg queries from the frame-0 anchor; if we DECAY
  qsupp late, bg queries re-attend the anchor's bg appearance -> re-ground style
  without a hard geometric tether (distinct from sink3's uniform hard tether and
  exp43's subject-boost ramp).
- code change: added `fg_qsupp_decay` / `fg_qsupp_floor`. In `maybe_apply`,
  qsupp_eff = max(floor, fg_qsupp*(1 - decay*start_frame)), passed to `_fgsink_soft`.
- params: robot NF=201, box(14,30,21,31), boost4, mid-gating, sink1/win6.
  Op-B: base qsupp3 + kb×2, decay {0,0.01,0.02} floor0.5.
  Op-A: base qsupp6 + kb×1, decay {0,0.01,0.02} floor2.5.
- outputs: `attn_injection_out/exp45_bgground/{ctrl_d0,dec01,dec02,A_ctrl_d0,
  A_dec01,A_dec02}/`; grids `_exp45.png`,`_exp45_zoom.png`,`_exp45A.png`,`_exp45A_zoom.png`.
- visual (full-res QA): **NEGATIVE result.** Op-B late bg is dominated by
  COHERENT approaching terrain (rock outcrop), not Link gear — decay changes it
  only marginally (nothing to remove; op-B already clean per exp44). Op-A: decay
  does NOT remove the green back-element — control vs dec02 at f100–130 are
  essentially identical.
- **SHARPENED diagnosis (the value of this round):** the residual green element is
  a Hylian-paraglider/shield the model composites IN THE SUBJECT REGION, attached
  to the humanoid's back (its humanoid+glider prior), NOT a free-background
  hallucination. Therefore: (i) background-side knobs (qsupp-decay) provably
  can't remove it; (ii) subject-side boost↑ would freeze the subject (exp40/43).
  It is a training-free-irreducible composition artifact. Onset ~f100–110, so the
  gear-free clean length is ≤~90–100f (consistent w/ exp41).
- next: subject identity+articulation+travel = SOLVED; residual is characterized
  as irreducible training-free. Deliver ≤~90–100f gear-free clips (op A, traveling)
  or op-B for longer-coherent stationary-ish world. qsupp-decay = DEAD END for the
  residual (keep the code; it's harmless and the floor=qsupp default disables it).

### exp44 (2026-06-21) — decouple subject motion (kb) from bg dynamism (qsupp)
- env/model/code: unchanged.
- params: robot NF=201, base recipe, LOW qsupp {2,3} + keyboard scale {2,3}.
- outputs: `attn_injection_out/exp44_decouple/{q2_kb2,q2_kb3,q3_kb2}/`; `_decouple.png`.
- visual: KEY — kb-scaling RESTORES subject articulation at low qsupp (q2_kb2 artic
  8.67 vs frozen ~1.5 w/o kb). So subject motion CAN come from the action drive, not
  only scene dynamism. Lower qsupp → lower departure (30 vs 67) → less haze/coherent
  snow. BUT Zelda creatures/figures STILL hallucinate in bg late (q3_kb2 yellow
  creature ~f190). Background hallucination = base-model OOD limit; its form changes
  but persists training-free.
- **TWO operating points:** (A) high qsupp = traveling world + articulating subject
  (bg hazes late); (B) low qsupp + kb×2-3 = coherent stationary-ish world +
  articulating subject (bg spawns creatures late). q3_kb2 a good (B).
- failure mode (FINAL): background Zelda-content hallucination over long OOD rollout
  is a base-model property; not removable training-free w/o retraining. Subject
  identity+articulation = SOLVED.
- next: subject side done + characterized; for a clean deliverable use clean length
  (~90-130f). Could try per-case (B)-style configs for the showreel. Diminishing
  returns on the bg residual without touching weights.
