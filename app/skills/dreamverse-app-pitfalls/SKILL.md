---
name: dreamverse-app-pitfalls
description: Hard-won pitfalls for developing & serving the FastVideo Output Video Workspace (app/) in this cluster environment — node location, login-vs-compute node split, how to actually view a dev server, tunnels, and screenshots. Read before doing dev/serve/debug work on app/.
---

# Output Video Workspace — pitfalls & how-to

A living log of non-obvious things that cost time. Append new lessons as you hit them;
keep each entry: **symptom → cause → fix**.

## Environment topology (the big one)

There are **two different machines**, and confusing them is the #1 time sink:

- **Compute node** `hpc-rack-2-2` (IP `10.244.14.252`): where the Claude/agent session
  runs, and where any server an agent starts will listen.
- **Login node** `slurm-login-*` (a k8s pod): where the **user's interactive terminal and
  VS Code / Cursor Remote-SSH are connected**.

Verified facts about this cluster:
- `/home` is **shared** across both nodes (so `app/` code + the `public/videos` symlink
  target under `/home/hal-kaiqin/FastVideo-Quantization/...` are visible on both).
- The login node **can reach** the compute node over the network
  (`curl http://10.244.14.252:5173` → `200`).

**Implication for serving a dev server (the decent, private way):**
Run `npm run dev` on the **login node** (where VS Code is connected), then use the IDE's
**Ports** panel to forward `5173`. Because `/home` is shared, the login-node vite serves
the exact same app + real videos. Do **not** start vite on the compute node and expect
`localhost` IDE/SSH forwarding to find it — IDE forwarding forwards the *login node's*
localhost, where nothing is listening.

```bash
# on the LOGIN node terminal (where your VS Code is):
export PATH="/home/hal-kaiqin/miniforge3/envs/FastVideo_kaiqin/bin:$PATH"
cd /home/hal-kaiqin/dreamverse/FastVideo_app/app
npm run dev          # then VS Code Ports -> forward 5173
```

## node / npm are not on PATH

- **Symptom:** `node: command not found`, `npm: command not found`.
- **Cause:** node lives inside the `FastVideo_kaiqin` conda env, not on the base PATH.
- **Fix:** prepend the env bin:
  `export PATH="/home/hal-kaiqin/miniforge3/envs/FastVideo_kaiqin/bin:$PATH"`
  (node v25.x, npm 11.x). The env is shared, so this works on both nodes.

## The `!` prefix is for the Claude chat box, NOT your SSH terminal

- **Symptom:** pasting `! curl ... && chmod ... && nohup ...` into the SSH terminal exits 1
  and the rest of the chain never runs.
- **Cause:** in a raw bash shell, a leading `!` is **logical-NOT** on the next pipeline.
  `! curl` inverts curl's success → exit 1 → the `&&` chain short-circuits.
- **Fix:** the `!`-prefix only means "run in the Claude session" when typed into the
  **Claude Code message box**. There it runs on the **compute node** (next to the agent),
  user-initiated, and `/tmp` is shared with the agent so it can read your logs. In your own
  SSH terminal, drop the `!`.

## Public tunnels (cloudflared / ngrok) are gated

- **Symptom:** the agent's attempts to download/run cloudflared get denied by the auto-mode
  classifier ("public exposure of real internal videos").
- **Cause:** `public/videos` symlinks real `FastVideo-Quantization` checkpoint outputs, so a
  public URL leaks real training data. The agent won't arm that automatically.
- **Fix:** prefer the **private** route above (login-node vite + IDE forward). If a public
  link is genuinely wanted, the **user** launches it (own terminal or chat `!`):
  - ngrok is pre-installed at `~/bin/ngrok` (aarch64, v3.39.x); token at
    `/home/hal-kaiqin/dreamverse/FastVideo-omninft/apps/dreamverse/ngrok.txt`.
  - From the login node, point the tunnel at the compute node:
    `~/bin/ngrok http 10.244.14.252:5173` (or run vite locally and use `5173`).
- **Vite gotcha:** a tunnel/forward sends a foreign `Host` header → Vite answers
  *"Blocked request. This host is not allowed."* Set `server.allowedHosts: true` in
  `vite.config.ts` (already set) and **restart** vite so it takes effect.

## `pkill -f cloudflared` can hit other people's processes

- **Symptom:** `pkill: killing pid <N> failed: Operation not permitted`.
- **Cause:** that PID is another user's / root's cloudflared on a shared node — not yours.
- **Fix:** kill only yours by matching the unique port:
  `pkill -f 'cloudflared.*5173'`. List first with `pgrep -af cloudflared`. Leave the
  `Operation not permitted` one alone.

## Killing a dev server without self-killing

- **Symptom:** `pkill -f vite` in a background agent shell SIGKILLs the shell itself (its own
  command line contains "vite").
- **Fix:** kill the npm parent by PID (kills the vite child via the process group), or:
  `for p in $(pgrep -x node); do grep -qa 'bin/vite' /proc/$p/cmdline && kill $p; done`
  (`pgrep -x node` won't match your bash). Confirm with `curl localhost:5173` → `000`.

## Headless Chrome also needs FONTS, or it aborts (not just libs)

- **Symptom:** Chrome/Playwright dies mid-render with `page closed` / SIGABRT;
  `chrome` stderr shows `FATAL:.../SkFontMgr_FontConfigInterface.cpp:163] Not implemented`
  and many `TextRunHarfBuzz error ... font: '' glyph_count: 0`. Looks like a "too many
  videos" crash but isn't.
- **Cause:** no fonts installed → Skia can't shape any text → fatal when a text-heavy view
  renders. (A page with little text may render; a richer one aborts — so it's intermittent.)
- **Fix:** install fonts into the throwaway prefix and point fontconfig at them:
  ```bash
  conda install -y -p /tmp/chromedeps -c conda-forge fonts-conda-ecosystem font-ttf-dejavu-sans-mono
  export FONTCONFIG_PATH=/tmp/chromedeps/etc/fonts FONTCONFIG_FILE=/tmp/chromedeps/etc/fonts/fonts.conf
  export LD_LIBRARY_PATH=/tmp/chromedeps/lib
  ```
  Also: many autoplaying `<video>` can still stress headless swiftshader — pass
  `--autoplay-policy=user-gesture-required` so cells show their `poster` instead of decoding.

## Stale data.json renders as a silently-empty UI

- **Symptom:** "all params" toggle (or any new metadata view) shows nothing; no console error.
- **Cause:** UI schema moved ahead of the generated `public/data.json` (e.g. UI reads
  `training_args` but data still has old `config`) — `?? {}` fallbacks turn the mismatch into
  empty renders. Happened when the Bash gate outage prevented `gen_demo_data.py` from running
  after a schema change shipped.
- **Fix / prevent:** always re-run `python3 tools/gen_demo_data.py` after generator/schema
  changes (then `build_static.sh`). The app now detects missing `training_args` and shows a
  red stale-schema banner instead of failing silently — keep that pattern for future schema
  bumps.

## Run dir names lie — derive run params from wandb args, never the name

- **Symptom:** runs named `...-ema0.99-lazy` vs `...-ema200` were labeled "ema 0.99 vs
  ema 200" by a dir-name regex — implying an absurd decay of 200.
- **Cause:** the name's "ema200" actually encodes `--ema-start-step 200`; the real arg-level
  diff (from each run's `tracker/wandb/latest-run/files/wandb-metadata.json` `args`) is
  `--ema-decay 0.99` present vs absent (→ TrainingArgs default 0.999). Both runs share
  `--ema-start-step 200`.
- **Fix:** parse the FULL CLI args from wandb-metadata.json and diff those; a missing arg
  renders as "default". Group params by FastVideo's own taxonomy — legacy stack
  `TrainingArgs` in `fastvideo/fastvideo_args.py:843-1273`, new stack dataclasses in
  `fastvideo/train/utils/training_config.py`. Fork-specific flags (e.g.
  `--generator_4bit_attn`) won't be in upstream TrainingArgs — bucket unknowns instead of
  dropping them. wandb-metadata.json also carries environment (gpu/cuda/host) and code
  provenance (git commit, program) worth surfacing.

## Svelte `each_key_duplicate` crashes at runtime, NOT in svelte-check

- **Symptom:** a view (e.g. the run-diff table) renders blank; pageerror
  `https://svelte.dev/e/each_key_duplicate`. `npm run check` is green.
- **Cause:** `{#each row.values as v (v)}` keyed by a value that repeats — identical cell
  values across runs (same seed/resolution/base model) collide.
- **Fix:** key by index/composite: `{#each row.values as v, i (i)}`. Only key by the value
  when it is guaranteed unique.

## pkill patterns can SIGTERM your own shell

- `pkill -f 'npm run preview'` (or `...vite`) matches the running shell's own command line
  (it contains that string) → kills itself, exit 144. Kill the vite/preview node child by
  PID via `pgrep -x node` + `/proc/$p/cmdline` check instead.

## Headless screenshots need system libs the box lacks

- **Symptom:** cached Playwright Chromium fails: `libnspr4.so: cannot open shared object`.
- **Cause:** no root to `apt install` the NSS/X11/gbm stack.
- **Fix:** install them into a **throwaway** conda prefix (do NOT mutate the shared env) and
  point `LD_LIBRARY_PATH` at it:
  ```bash
  conda create -y -p /tmp/chromedeps -c conda-forge nspr nss atk-1.0 at-spi2-core dbus \
    xorg-libx11 xorg-libxcomposite xorg-libxdamage xorg-libxext xorg-libxfixes \
    xorg-libxrandr libxcb libxkbcommon libgbm libdrm expat cairo pango libcups
  LD_LIBRARY_PATH=/tmp/chromedeps/lib node screenshot.mjs   # playwright-core, executablePath=cached chrome
  ```

## NEVER run two Vite dev servers from the same (shared) node_modules

- **Symptom:** the app was working, then "suddenly won't open" even after restarting vite.
- **Cause:** `/home` is shared, so the agent's compute-node vite and the user's login-node
  vite ran from the **same** `node_modules`. Two Vite dev servers race on the shared
  dependency-optimize cache `node_modules/.vite` and corrupt it; the bad cache persists on
  disk, so a plain restart stays broken.
- **Fix:** `rm -rf node_modules/.vite` then `npm run dev -- --force` to rebuild it clean.
- **Prevent:** the agent must NOT start a second vite for verification on the compute node
  while the user has one on the login node. Verify on a *different port AND* avoid relying
  on the shared optimize cache, or just don't — drive the user's running instance instead.

## `+` in URL query params decodes to SPACE — never use it in shareable tokens

- **Symptom:** deep link `#/results?runs=exp_37+exp_38` renders empty (token lookup fails);
  in-app navigation works (URLSearchParams encodes `+` as `%2B`).
- **Cause:** per URL spec, a literal `+` in a query string decodes to a space.
- **Fix:** use `~` (URL-safe, no decoding ambiguity) as the merged-run token separator.
  Also note: multi-line `token\n.split("+")` escapes a single-line `sed` — grep the built
  bundle (`grep -o 'split("[+~]")' dist/assets/*.js`) to confirm what actually shipped.

## Real-scale ingestion (51 runs / ~7000 videos) — what broke and the fixes

- **Validation sets must be namespaced.** Different runs use different validation files
  (`validation.json` vs `validation_16.json`) with different prompts at the same video
  index. A global `case_{idx}` key silently mixes prompts across sets. Fix: case id =
  `case_{vset}_{idx}` (vset = sanitized validation-file stem); experiments carry
  `validation_set`; compare views filter rows to the selected runs' sets.
- **Thumbnails must be parallel.** ~7000 ffmpeg first-frame extractions serially ≈ 15 min;
  `ThreadPoolExecutor(max_workers=16)` → 51 s wall (cache makes re-runs instant).
- **Default compare set must be empty at scale.** With every ingested run flagged
  `in_grid`, "default = in_grid" would mean a 51-column grid. Default selection = [] and
  the graph is the picker.
- **Graph columns must wrap.** ~30 root runs (no lineage parents) stack into one
  unreadably tall depth-column; fit-view then zooms to confetti. Wrap each depth column
  into sub-columns of ≤8 rows (mosaic) — landscape, readable at fit zoom.
- **Steps differ across arbitrary selections** (resume runs start at 2600+). Pure
  intersection of steps can be empty → use intersection, fall back to union (cells render
  "not generated" where a run lacks that step).
- Discovery flag: `python3 tools/gen_demo_data.py --ckpt-root <checkpoints-dir>` ingests
  every run dir with wandb metadata (skips `*_weight_only`).

## Performance: don't eagerly mount every cell's `<video>`

- **Symptom:** the flat matrix loads extremely slowly.
- **Cause:** rendering all cells = N_cases × N_checkpoints `<video preload="metadata">`
  elements fires hundreds of header fetches at once; it does not scale as runs grow.
- **Fix:** see the matrix redesign (poster thumbnails + lazy `<video>` via
  `IntersectionObserver` + progressive disclosure / windowing). Grid cells should be cheap
  `<img>` posters; instantiate a real `<video>` only on demand.
