#!/usr/bin/env python3
import argparse
import json
import os
import re
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

DEFAULT_CKPT_DIRS = [
    "/home/hal-kaiqin/FastVideo-Quantization/checkpoints/wan_dmd_distill_20260605_2304_dmd_4bATTN-4bLIN-realFT-realDMD-cfg2.0-ema0.99-lazy",
    "/home/hal-kaiqin/FastVideo-Quantization/checkpoints/wan_dmd_distill_20260605_1042_dmd_4bATTN-4bLIN-realFT-realDMD-cfg2.0-ema200",
]

VIDEO_RE = re.compile(r"validation_step_(\d+)_inference_steps_(\d+)_video_(\d+)\.mp4$")
LINEAGE_RE = re.compile(r"checkpoints/([^/]+)/checkpoint-(\d+)")
LINEAGE_ARGS = [("init_weights_from_safetensors", "init_weights"),
                ("resume_from_checkpoint", "resume")]


def load_wandb_meta(ckpt_dir):
    meta = ckpt_dir / "tracker" / "wandb" / "latest-run" / "files" / "wandb-metadata.json"
    if not meta.exists():
        return {}
    return json.loads(meta.read_text())


def parse_args_list(raw):
    out = {}
    i = 0
    while i < len(raw):
        token = raw[i]
        if isinstance(token, str) and token.startswith("--"):
            key = token[2:]
            if i + 1 < len(raw) and not str(raw[i + 1]).startswith("--"):
                out[key] = raw[i + 1]
                i += 2
            else:
                out[key] = True
                i += 1
        else:
            i += 1
    return out


def parse_environment(meta):
    gpus = meta.get("gpu_nvidia") or []
    return prune({
        "gpu": meta.get("gpu"),
        "gpu_count": meta.get("gpu_count"),
        "gpu_arch": gpus[0].get("architecture") if gpus else None,
        "cuda": meta.get("cudaVersion"),
        "python": meta.get("python"),
        "os": meta.get("os"),
        "host": meta.get("host"),
    })


def parse_code(meta):
    git = meta.get("git") or {}
    return prune({
        "git_remote": git.get("remote"),
        "git_commit": git.get("commit"),
        "program": meta.get("codePath") or meta.get("program"),
        "started_at": meta.get("startedAt"),
    })


def find_validation_file(name, ckpt_dir):
    candidates = [
        Path("/home/hal-kaiqin/FastVideo-Quantization") / name,
        ckpt_dir.parent.parent / name,
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def vset_id(args):
    rel = args.get("validation_dataset_file")
    if not rel:
        return None
    stem = Path(str(rel)).stem
    return re.sub(r"\W+", "_", stem)


def load_captions(args, ckpt_dir):
    rel = args.get("validation_dataset_file")
    if not rel:
        return {}
    path = find_validation_file(rel, ckpt_dir)
    if path is None:
        return {}
    data = json.loads(path.read_text())
    rows = data.get("data", data) if isinstance(data, dict) else data
    return {idx: (row.get("caption") or row.get("prompt") or "") for idx, row in enumerate(rows)}


def collect_videos(ckpt_dir):
    found = []
    for p in sorted(ckpt_dir.glob("*.mp4")):
        m = VIDEO_RE.search(p.name)
        if m:
            found.append({
                "step": int(m.group(1)),
                "inference_steps": int(m.group(2)),
                "video_idx": int(m.group(3)),
                "name": p.name,
            })
    return found


def make_thumb(src, dst, ffmpeg):
    if dst.exists():
        return True
    if not ffmpeg:
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        r = subprocess.run(
            [ffmpeg, "-y", "-loglevel", "error", "-i", str(src),
             "-frames:v", "1", "-vf", "scale=320:-1", str(dst)],
            timeout=60,
        )
        return r.returncode == 0 and dst.exists()
    except Exception:
        return False


def make_thumbs_parallel(jobs, ffmpeg, workers=16):
    if not jobs:
        return
    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(lambda j: make_thumb(j[0], j[1], ffmpeg), jobs))


def wandb_run_id(ckpt_dir):
    runs = ckpt_dir / "tracker" / "wandb"
    if not runs.exists():
        return None
    for d in runs.iterdir():
        if d.name.startswith("run-"):
            return d.name.split("-")[-1]
    return None


def to_num(v, cast):
    try:
        return cast(v)
    except (TypeError, ValueError):
        return v


def prune(d):
    return {k: v for k, v in d.items() if v is not None}


def get_or_add(index, key, arr, factory):
    if key in index:
        return index[key]
    obj = factory(len(arr))
    arr.append(obj)
    index[key] = obj["id"]
    return obj["id"]


def build(ckpt_dirs, out_path, gen_thumbs=True):
    ckpt_dirs = [Path(d) for d in ckpt_dirs]
    out_path = Path(out_path)
    thumbs_dir = out_path.parent / "thumbs"
    ffmpeg = shutil.which("ffmpeg") if gen_thumbs else None
    if gen_thumbs and not ffmpeg:
        print("warning: ffmpeg not found on PATH — posters will be empty")

    if len(ckpt_dirs) == 1:
        common = ckpt_dirs[0].parent
    else:
        common = Path(os.path.commonpath([str(d) for d in ckpt_dirs]))

    projects, pipelines, base_models, datasets = [], [], [], []
    experiments, checkpoints, output_videos = [], [], []
    cases_by_key = {}
    proj_ids, pipe_ids, bm_ids, ds_ids = {}, {}, {}, {}
    exp_by_run_name = {}
    raw_lineage = []
    thumb_jobs = []
    ckpt_root = ckpt_dirs[0].parent

    def ingest(ckpt_dir, in_grid):
        e = len(experiments)
        meta = load_wandb_meta(ckpt_dir)
        args = parse_args_list(meta.get("args", []))
        name = ckpt_dir.name

        proj_name = args.get("tracker_project_name", "unknown_project")
        proj_id = get_or_add(proj_ids, proj_name, projects,
                             lambda i: {"id": f"proj_{i}", "name": proj_name})
        program = meta.get("codePath", "") or ""
        pipe_kind = "dmd_distill" if "distill" in program else "finetune"
        pipe_name = "DMD distillation" if pipe_kind == "dmd_distill" else "Finetune"
        pipe_id = get_or_add(pipe_ids, f"{proj_id}|{pipe_kind}", pipelines,
                             lambda i: {"id": f"pipe_{i}", "project_id": proj_id,
                                        "name": pipe_name, "kind": pipe_kind})
        bm_repo = args.get("model_path", "unknown")
        bm_id = get_or_add(bm_ids, bm_repo, base_models,
                           lambda i: {"id": f"bm_{i}", "name": bm_repo.split("/")[-1], "hf_repo": bm_repo})
        ds_path = args.get("data_path", "unknown")
        p = Path(ds_path)
        ds_name = (p.parent.name if p.name in ("combined_parquet_dataset", "dataset", "data") and p.parent.name
                   else p.name) or ds_path
        ds_id = get_or_add(ds_ids, ds_path, datasets,
                           lambda i: {"id": f"ds_{i}", "name": ds_name, "path": ds_path})

        resolution = None
        if args.get("num_width") and args.get("num_height"):
            resolution = f"{args['num_height']}x{args['num_width']}"
        guidance = to_num(args.get("validation_guidance_scale"), float)

        exp_id = f"exp_{e}"
        vset = vset_id(args)
        experiments.append({
            "id": exp_id,
            "name": name,
            "project_id": proj_id,
            "pipeline_id": pipe_id,
            "base_model_id": bm_id,
            "dataset_id": ds_id,
            "init_from": args.get("init_weights_from_safetensors", ""),
            "wandb_run_id": wandb_run_id(ckpt_dir),
            "in_grid": in_grid,
            "validation_set": vset,
            "training_args": args,
            "environment": parse_environment(meta),
            "code": parse_code(meta),
        })
        exp_by_run_name[name] = exp_id

        for arg_key, kind in LINEAGE_ARGS:
            m = LINEAGE_RE.search(str(args.get(arg_key, "")))
            if m:
                raw_lineage.append({"child": exp_id, "parent_name": m.group(1),
                                    "kind": kind, "step": int(m.group(2))})

        if not in_grid:
            return

        captions = load_captions(args, ckpt_dir)
        videos = collect_videos(ckpt_dir)
        vkey = vset or "none"

        for s in sorted({v["step"] for v in videos}):
            checkpoints.append({
                "id": f"ckpt_{e}_{s}",
                "experiment_id": exp_id,
                "step": s,
                "path": str(ckpt_dir / f"checkpoint-{s}") if s > 0 else str(ckpt_dir),
            })

        for vi in sorted({v["video_idx"] for v in videos}):
            if (vkey, vi) not in cases_by_key:
                cap = captions.get(vi, "")
                cases_by_key[(vkey, vi)] = {
                    "id": f"case_{vkey}_{vi}",
                    "index": vi,
                    "caption": cap if cap else f"{vkey} #{vi:02d}",
                    "has_real_caption": bool(cap),
                    "validation_set": vkey,
                    "resolution": resolution,
                }

        for v in videos:
            ov_id = f"ov_{e}_{v['step']}_{v['video_idx']}_{v['inference_steps']}"
            rel = os.path.relpath(str(ckpt_dir / v["name"]), str(common))
            jpg = thumbs_dir / f"{ov_id}.jpg"
            thumb_jobs.append((ckpt_dir / v["name"], jpg))
            output_videos.append({
                "id": ov_id,
                "experiment_id": exp_id,
                "checkpoint_id": f"ckpt_{e}_{v['step']}",
                "validation_case_id": f"case_{vkey}_{v['video_idx']}",
                "sampling": prune({
                    "inference_steps": v["inference_steps"],
                    "guidance_scale": guidance,
                    "seed": to_num(args.get("seed"), int),
                    "fps": 16,
                }),
                "file_path": f"videos/{rel}",
                "poster_path": f"thumbs/{ov_id}.jpg",
                "exists": True,
            })

    for ckpt_dir in ckpt_dirs:
        ingest(ckpt_dir, in_grid=True)

    while True:
        pending = [l["parent_name"] for l in raw_lineage
                   if l["parent_name"] not in exp_by_run_name
                   and (ckpt_root / l["parent_name"]).is_dir()]
        if not pending:
            break
        for parent_name in dict.fromkeys(pending):
            ingest(ckpt_root / parent_name, in_grid=False)

    lineage = [
        {"from": exp_by_run_name[l["parent_name"]], "to": l["child"],
         "kind": l["kind"], "step": l["step"]}
        for l in raw_lineage if l["parent_name"] in exp_by_run_name
    ]

    print(f"  generating thumbnails: {len(thumb_jobs)} jobs (cached ones skipped)…")
    make_thumbs_parallel(thumb_jobs, ffmpeg)
    for ov in output_videos:
        if ov["poster_path"] and not (out_path.parent / ov["poster_path"]).exists():
            ov["poster_path"] = ""

    doc = {
        "projects": projects,
        "pipelines": pipelines,
        "base_models": base_models,
        "dataset_snapshots": datasets,
        "experiments": experiments,
        "checkpoints": checkpoints,
        "validation_cases": [cases_by_key[k] for k in sorted(cases_by_key)],
        "output_videos": output_videos,
        "lineage": lineage,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(doc, indent=2, ensure_ascii=False))
    return doc, common


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", nargs="+", default=None)
    ap.add_argument("--ckpt-root", default=None,
                    help="ingest every run dir under this root that has wandb metadata")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent.parent / "public" / "data.json"))
    ap.add_argument("--no-symlink", action="store_true")
    ap.add_argument("--no-thumbs", action="store_true")
    a = ap.parse_args()

    if a.ckpt_root:
        root = Path(a.ckpt_root)
        dirs = sorted(
            str(d) for d in root.iterdir()
            if d.is_dir() and not d.name.endswith("_weight_only")
            and (d / "tracker/wandb/latest-run/files/wandb-metadata.json").exists()
        )
    else:
        dirs = a.ckpt_dir or DEFAULT_CKPT_DIRS

    doc, common = build(dirs, a.out, gen_thumbs=not a.no_thumbs)

    if not a.no_symlink:
        link = Path(a.out).parent / "videos"
        if link.is_symlink() or link.exists():
            if link.is_symlink():
                link.unlink()
        if not link.exists():
            os.symlink(common, link)

    n_posters = sum(1 for v in doc["output_videos"] if v.get("poster_path"))
    print(f"wrote {a.out}")
    print(f"  experiments={len(doc['experiments'])} checkpoints={len(doc['checkpoints'])} "
          f"cases={len(doc['validation_cases'])} output_videos={len(doc['output_videos'])} "
          f"(posters={n_posters})")
    print(f"  videos symlink target: {common}")
    vsets = {}
    for c in doc["validation_cases"]:
        vsets[c["validation_set"]] = vsets.get(c["validation_set"], 0) + 1
    print(f"  validation sets: {vsets}")
    print(f"  lineage edges: {len(doc['lineage'])}")


if __name__ == "__main__":
    main()
