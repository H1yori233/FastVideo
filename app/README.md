# Output Video Workspace

A lightweight **video result workspace** for FastVideo training runs — *not* a run
tracker. The central object is the **OutputVideo**: you browse validation videos first,
then trace each one back to its experiment → checkpoint → validation case → base model →
dataset.

- Main view: **Output Video Matrix** — rows = validation cases, columns = (experiment,
  checkpoint-step) pairs, cells = output videos. Missing/not-yet-generated videos show a
  clean placeholder.
- Click a cell → **detail panel** with the video plus its full provenance.

Stack: Vite + React 19 + TypeScript + Tailwind CSS v4 + shadcn-style components + motion
(framer-motion). Data is local JSON; light user edits (graph layout) live in
`localStorage`. No backend, no database, no auth.

Views (graph-first):
- **Graph (home)** — the management hub: every run with auto-derived lineage
  (`init_weights_from_safetensors` / `resume_from_checkpoint`). Search runs, pan/zoom,
  drag nodes (layout persists), click a node for its full configuration + lineage links,
  and **select runs (○/◉) to build a comparison set** — Compare N runs →.
- **Results** — figure list for the selected runs: each validation case as a serif-captioned
  figure with one video plate per run (posters only, 0 video downloads).
- **Case** — side-by-side players + Table 1 (differing hyperparameters, honest `default`
  for unset args) + Appendix (full grouped configuration).

## Run

Node is not on the default PATH in this environment; it lives in the `FastVideo_kaiqin`
conda env. Put it on PATH first:

```bash
export PATH="/home/hal-kaiqin/miniforge3/envs/FastVideo_kaiqin/bin:$PATH"
cd /home/hal-kaiqin/dreamverse/FastVideo_app/app

python3 tools/gen_demo_data.py   # generate public/data.json + public/videos symlink
npm install
npm run dev                      # http://localhost:5173
```

## Demo data

`tools/gen_demo_data.py` scans a real training checkpoint directory and emits
`public/data.json`, plus a `public/videos` symlink so Vite serves the mp4s without copying
them. Point it at a different run with `--ckpt-dir`:

```bash
python3 tools/gen_demo_data.py --ckpt-dir /path/to/checkpoints/<run-name>
```

It recovers provenance from the run's wandb metadata (base model, dataset, init-from
lineage, project, resolution/frames/seed) and validation prompts from the run's
`validation_dataset_file`. Cases without a prompt fall back to `Val-NN`. One extra
"queued" checkpoint column with `exists=false` videos is injected to exercise the
placeholder path.

## Layout

```
src/lib/types.ts                 schema (TS interfaces, incl. LineageEdge)
src/lib/dataStore.ts             Workspace: load data.json, indexes, compare/lineage helpers
src/lib/paramGroups.ts           FastVideo TrainingArgs taxonomy (grouping for params/diff)
src/components/
  RunCompareGrid.tsx             cases × runs poster grid + step selector
  CaseCompareView.tsx            side-by-side players + grouped args diff
  RunGraph.tsx                   lineage canvas (SVG, pan/zoom/drag)
  RunCard.tsx                    run card (compact / full with grouped params)
  VideoTile.tsx, useInView.ts    poster tile / in-view autoplay
  ui/                            shadcn-style primitives (badge, button, tabs, select)
tools/gen_demo_data.py           data generator (multi-run + ancestor/lineage discovery)
tools/serve.sh, check_deploy.sh  one-command serve / deployment health check
public/data.json                 generated
public/videos                    generated symlink -> checkpoints root
public/thumbs/                   generated posters (ffmpeg first frames)
```

## Roadmap (next slices)

- **Curation:** mark videos interesting / candidate / broken, notes + tags (localStorage).
- **Filter bar:** by status / run / case caption search.
- **Run picker:** choose which runs join the compare grid from the graph.
