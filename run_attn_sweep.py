#!/usr/bin/env python3
"""
Sweep local_attn_size and sink_size for Matrix-Game (Skyreels Solaris) inference.

- local_attn_size: temporal sliding window (in frames) for causal KV attention;
  smaller = less temporal context / lower memory, larger = more history.
- sink_size: number of initial frames kept as attention sink when rolling KV;
  0 = no fixed prefix; larger = more “anchor” tokens from the start.

Each run uses a temporary model directory: symlinks to the real weights plus a
modified transformer/config.json (local_attn_size / sink_size).

Runs execute in order: for each local_attn_size, for each sink_size (e.g. 8
runs: la6×sk0..3, la9×sk0..3). After all succeed, ffmpeg xstacks the
eight clips into one 4×2 montage (montage_4x2.mp4 under --output-root).
Each tile is overlaid with a label (e.g. la6_sk0) matching the run folder name.

Usage (from repo root or FastVideo dir, with conda env active):
  python run_attn_sweep.py
  python run_attn_sweep.py --dry-run
  python run_attn_sweep.py --local-attn-sizes 6 --sink-sizes 0 1
  python run_attn_sweep.py --no-grid-montage   # skip ffmpeg step
  python run_attn_sweep.py --montage-only      # only ffmpeg montage from existing la*_sk*/output.mp4
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# Ensure FastVideo package is importable when run as `python run_attn_sweep.py`
_FASTVIDEO_ROOT = Path(__file__).resolve().parent
if str(_FASTVIDEO_ROOT) not in sys.path:
    sys.path.insert(0, str(_FASTVIDEO_ROOT))


DEFAULT_MODEL = "/home/hal-kaiqin/models/skyreels_solaris"
# DEFAULT_MODEL = "/home/hal-kaiqin/models/mg_solaris"
DEFAULT_IMG = "/home/hal-kaiqin/models/image.png"
# DEFAULT_OUT_ROOT = "video_samples_attn_sweep"
DEFAULT_OUT_ROOT = "video_samples_attn_sweep_inf.rope"


def _ffprobe_bin() -> str:
    """Resolve ffprobe: PATH, or same directory as ffmpeg (common in conda envs)."""
    found = shutil.which("ffprobe")
    if found:
        return found
    ffm = shutil.which("ffmpeg")
    if ffm:
        cand = os.path.join(os.path.dirname(ffm), "ffprobe")
        if os.path.isfile(cand):
            return cand
    return "ffprobe"


def build_model_dir_with_config(
    source_root: str,
    local_attn_size: int,
    sink_size: int,
) -> str:
    """Symlink-copy model tree; write a real transformer/config.json override."""
    source_root = os.path.abspath(source_root)
    tmp = tempfile.mkdtemp(prefix="matrixgame_attn_")
    for name in os.listdir(source_root):
        if name.startswith("."):
            continue
        src = os.path.join(source_root, name)
        dst = os.path.join(tmp, name)
        if name == "transformer":
            os.makedirs(dst, exist_ok=True)
            for sub in os.listdir(src):
                ssub = os.path.join(src, sub)
                dsub = os.path.join(dst, sub)
                if sub == "config.json":
                    with open(ssub, encoding="utf-8") as f:
                        cfg = json.load(f)
                    cfg["local_attn_size"] = int(local_attn_size)
                    cfg["sink_size"] = int(sink_size)
                    with open(dsub, "w", encoding="utf-8") as f:
                        json.dump(cfg, f, indent=4)
                        f.write("\n")
                else:
                    os.symlink(ssub, dsub)
        else:
            os.symlink(src, dst)
    return tmp


def _video_size_imageio(path: str) -> tuple[int, int]:
    """Decode first frame via imageio (no ffprobe required)."""
    import imageio.v3 as iio

    frame = iio.imread(path, index=0)
    h, w = int(frame.shape[0]), int(frame.shape[1])
    return w, h


def _verify_montage_output_shape(
    out_path: str,
    expected_w: int,
    expected_h: int,
) -> None:
    """Decode the written file's first frame and assert (W,H) matches expectation.

    This is the ground-truth check: actual decoded ndarray shape, not ffmpeg logs.
    """
    import imageio.v3 as iio

    if not os.path.isfile(out_path) or os.path.getsize(out_path) == 0:
        raise RuntimeError(f"Montage output missing or empty: {out_path}")

    frame = iio.imread(out_path, index=0)
    sh = frame.shape
    if len(sh) < 2:
        raise RuntimeError(f"Unexpected frame shape for {out_path}: {sh}")
    h, w = int(sh[0]), int(sh[1])
    print(
        "Montage output check (decoded first frame): "
        f"ndarray.shape={sh} (H,W,...) -> width={w}, height={h} "
        f"(expected {expected_w}x{expected_h})"
    )
    if w != expected_w or h != expected_h:
        raise RuntimeError(
            f"Montage size mismatch: decoded {w}x{h} != expected "
            f"{expected_w}x{expected_h}; shape={sh}"
        )


def _ffprobe_video_size(path: str) -> tuple[int, int]:
    """Return (width, height) of the first video stream."""
    cmd = [
        _ffprobe_bin(),
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height",
        "-of",
        "csv=p=0",
        path,
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True)
    except FileNotFoundError:
        proc = None
    if proc is not None and proc.returncode == 0 and proc.stdout:
        line = proc.stdout.strip().splitlines()[-1]
        for sep in ("x", ","):
            if sep in line:
                a, b = line.split(sep, 1)
                return int(a.strip()), int(b.strip())
    return _video_size_imageio(path)


def ffmpeg_xstack_montage(
    video_paths: list[str],
    out_path: str,
    cols: int,
    rows: int,
    labels: list[str] | None = None,
) -> None:
    """Concatenate N videos of equal length into a cols×rows grid (row-major order).

    Each input is scaled+padded to the **same** tile size (from the first clip's
    resolution), so the output frame is **cols × tile_w** by **rows × tile_h**.

    If ``labels`` is provided (len == N), each tile gets a top-left ``drawtext`` overlay.
    """
    n = len(video_paths)
    if n != cols * rows:
        raise ValueError(
            f"Need cols*rows ({cols}*{rows}={cols * rows}) videos, got {n}"
        )
    if labels is not None and len(labels) != n:
        raise ValueError(f"Need {n} labels, got {len(labels)}")
    for p in video_paths:
        if not os.path.isfile(p):
            raise FileNotFoundError(p)

    w0, h0 = _ffprobe_video_size(video_paths[0])
    # Even dimensions for yuv420p / libx264
    tw, th = (w0 // 2) * 2, (h0 // 2) * 2
    if tw < 2 or th < 2:
        raise ValueError(f"Invalid tile size {tw}x{th} from first input")

    # Normalize every input to tw×th so xstack canvas = cols*tw × rows*th
    norm_parts = [
        f"[{i}:v]scale={tw}:{th}:force_original_aspect_ratio=decrease,"
        f"pad={tw}:{th}:(ow-iw)/2:(oh-ih)/2[ti{i}]"
        for i in range(n)
    ]

    # xstack layout: absolute pixel (x,y) top-left of each tile — not w0/h0
    # expressions (those are easy to mis-parse and produced wrong canvas size).
    layout_parts: list[str] = []
    for i in range(n):
        col, row = i % cols, i // cols
        x_pos = col * tw
        y_pos = row * th
        layout_parts.append(f"{x_pos}_{y_pos}")
    layout = "|".join(layout_parts)

    if labels:
        def _safe_label(s: str) -> str:
            if not all(c.isalnum() or c in "_=-" for c in s):
                raise ValueError(f"Unsafe montage label for ffmpeg: {s!r}")
            return s

        dt_parts: list[str] = []
        for i in range(n):
            t = _safe_label(labels[i])
            dt_parts.append(
                f"[ti{i}]drawtext=text='{t}':fontsize=22:fontcolor=white:"
                f"x=10:y=10:box=1:boxcolor=black@0.55:borderw=2:bordercolor=black[lb{i}]"
            )
        stack_inputs = "".join(f"[lb{i}]" for i in range(n))
        filter_complex = (
            ";".join(norm_parts)
            + ";"
            + ";".join(dt_parts)
            + f";{stack_inputs}xstack=inputs={n}:layout={layout}:fill=black[xs];"
            + "[xs]scale=trunc(iw/2)*2:trunc(ih/2)*2[outv]"
        )
    else:
        stack_inputs = "".join(f"[ti{i}]" for i in range(n))
        filter_complex = (
            ";".join(norm_parts)
            + f";{stack_inputs}xstack=inputs={n}:layout={layout}:fill=black[xs];"
            + "[xs]scale=trunc(iw/2)*2:trunc(ih/2)*2[outv]"
        )

    out_w, out_h = cols * tw, rows * th
    print(
        f"Montage: tile {tw}x{th} (from first clip {w0}x{h0}), "
        f"output {out_w}x{out_h}"
    )

    ffmpeg_exe = shutil.which("ffmpeg") or "ffmpeg"
    cmd: list[str] = [ffmpeg_exe, "-y"]
    for p in video_paths:
        cmd.extend(["-i", p])
    cmd.extend(
        [
            "-filter_complex",
            filter_complex,
            "-map",
            "[outv]",
            "-an",
            "-c:v",
            "libx264",
            "-crf",
            "18",
            "-preset",
            "medium",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            out_path,
        ]
    )
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(
            proc.returncode, cmd, proc.stdout, proc.stderr
        )
    _verify_montage_output_shape(out_path, out_w, out_h)
    # Optional cross-check: ffprobe stream metadata (no imageio fallback)
    try:
        proc2 = subprocess.run(
            [
                _ffprobe_bin(),
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height",
                "-of",
                "csv=p=0",
                out_path,
            ],
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        pass
    else:
        if proc2.returncode == 0 and proc2.stdout:
            line = proc2.stdout.strip().splitlines()[-1]
            for sep in ("x", ","):
                if sep in line:
                    a, b = line.split(sep, 1)
                    print(
                        "Montage output check (ffprobe stream): "
                        f"{int(a.strip())}x{int(b.strip())}"
                    )
                    break


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Grid search local_attn_size × sink_size for Solaris Matrix-Game."
    )
    p.add_argument(
        "--model-path",
        default=DEFAULT_MODEL,
        help="Directory containing skyreels_solaris (default: %(default)s)",
    )
    p.add_argument(
        "--output-root",
        default=DEFAULT_OUT_ROOT,
        help="Directory to store subfolders per (local_attn, sink) run",
    )
    p.add_argument(
        "--local-attn-sizes",
        type=int,
        nargs="+",
        default=[6, 9],
        help="local_attn_size values (default: 6 9)",
    )
    p.add_argument(
        "--sink-sizes",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3],
        help="sink_size values (default: 0 1 2 3)",
    )
    p.add_argument("--image-path", default=DEFAULT_IMG)
    p.add_argument("--num-frames", type=int, default=1677)
    p.add_argument("--height", type=int, default=352)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--num-inference-steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--keyboard-dim", type=int, default=6)
    p.add_argument("--keyboard-index", type=int, default=13)
    p.add_argument(
        "--keep-temp-dirs",
        action="store_true",
        help="Do not delete symlink temp model dirs after each run",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print grid only; do not load models or generate",
    )
    p.add_argument(
        "--no-grid-montage",
        action="store_true",
        help="Do not ffmpeg xstack 8 clips into one 4x2 montage after sweep",
    )
    p.add_argument(
        "--no-montage-labels",
        action="store_true",
        help="Montage without per-tile drawtext (la*_sk*); use if ffmpeg/libfreetype fails",
    )
    p.add_argument(
        "--grid-cols",
        type=int,
        default=4,
        help="Montage columns (default 4 for 4x2)",
    )
    p.add_argument(
        "--grid-rows",
        type=int,
        default=2,
        help="Montage rows (default 2 for 4x2)",
    )
    p.add_argument(
        "--montage-name",
        default="montage_4x2.mp4",
        help="Filename under output-root for the combined video",
    )
    p.add_argument(
        "--montage-only",
        action="store_true",
        help=(
            "Skip inference; only build montage from existing files under "
            "output-root/la*_sk*/ (uses --local-attn-sizes × --sink-sizes order). "
            "Requires ffmpeg in PATH."
        ),
    )
    p.add_argument(
        "--montage-input",
        default="output.mp4",
        help="Video filename inside each la*_sk* folder (default: %(default)s)",
    )
    return p.parse_args()


def _try_ffmpeg_montage(
    args: argparse.Namespace,
    out_root: str,
    montage_paths: list[str],
    montage_labels: list[str],
    expected_count: int,
) -> None:
    """Run xstack if counts match grid and ``--no-grid-montage`` is not set."""
    if args.no_grid_montage:
        return
    if len(montage_paths) != expected_count:
        print(
            f"Montage skipped: only {len(montage_paths)}/{expected_count} inputs.",
            file=sys.stderr,
        )
        return
    if expected_count != args.grid_cols * args.grid_rows:
        print(
            f"Montage skipped: grid {args.grid_cols}x{args.grid_rows} "
            f"!= {expected_count} runs.",
            file=sys.stderr,
        )
        return
    montage_out = os.path.join(out_root, args.montage_name)
    try:
        ffmpeg_xstack_montage(
            montage_paths,
            montage_out,
            cols=args.grid_cols,
            rows=args.grid_rows,
            labels=None if args.no_montage_labels else montage_labels,
        )
        print(f"Montage ({args.grid_cols}x{args.grid_rows}): {montage_out}")
    except FileNotFoundError as e:
        print(
            "Montage skipped: missing video file or ffmpeg not in PATH.",
            e,
            file=sys.stderr,
        )
    except subprocess.CalledProcessError as e:
        err = getattr(e, "stderr", None) or getattr(e, "stdout", None) or str(e)
        print("ffmpeg failed:", err, file=sys.stderr)


def run_montage_only(args: argparse.Namespace) -> None:
    """Assemble grid from existing ``output-root/la*_sk*/`` clips (no GPU)."""
    combos = [
        (la, sk)
        for la in args.local_attn_sizes
        for sk in args.sink_sizes
    ]
    n = len(combos)
    print(
        f"Montage-only: {n} tiles — local_attn {args.local_attn_sizes}, "
        f"sink {args.sink_sizes}"
    )
    if n != args.grid_cols * args.grid_rows:
        print(
            f"Error: need grid_cols*grid_rows == {n}, got "
            f"{args.grid_cols}*{args.grid_rows}={args.grid_cols * args.grid_rows}.",
            file=sys.stderr,
        )
        sys.exit(1)
    out_root = os.path.abspath(args.output_root)
    montage_paths: list[str] = []
    montage_labels: list[str] = []
    for la, sk in combos:
        tag = f"la{la}_sk{sk}"
        rel = os.path.join(tag, args.montage_input)
        path = os.path.join(out_root, rel)
        if not os.path.isfile(path):
            print(f"Error: missing file for {tag}: {path}", file=sys.stderr)
            sys.exit(1)
        montage_paths.append(path)
        montage_labels.append(tag)
        print(f"  {tag}  <-  {path}")
    _try_ffmpeg_montage(
        args,
        out_root,
        montage_paths,
        montage_labels,
        n,
    )


def main() -> None:
    args = parse_args()
    combos = [
        (la, sk)
        for la in args.local_attn_sizes
        for sk in args.sink_sizes
    ]
    print(
        f"Grid: {len(combos)} runs — local_attn_size in {args.local_attn_sizes}, "
        f"sink_size in {args.sink_sizes}"
    )
    if args.dry_run:
        for la, sk in combos:
            print(f"  local_attn_size={la}  sink_size={sk}")
        if args.montage_only:
            root = os.path.abspath(args.output_root)
            print("Montage-only would read:")
            for la, sk in combos:
                tag = f"la{la}_sk{sk}"
                print(f"  {os.path.join(root, tag, args.montage_input)}")
        return

    if args.montage_only:
        run_montage_only(args)
        return

    import torch
    from fastvideo import VideoGenerator
    from fastvideo.models.dits.matrixgame.utils import create_action_presets

    out_root = os.path.abspath(args.output_root)
    os.makedirs(out_root, exist_ok=True)
    summary_path = os.path.join(
        out_root, f"sweep_summary_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    )
    grid_sizes = torch.tensor([420, 44, 80])
    # Same order as `combos`: local_attn outer, sink inner — sequential runs
    montage_paths: list[str] = []
    montage_labels: list[str] = []

    with open(summary_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            [
                "local_attn_size",
                "sink_size",
                "seconds",
                "peak_memory_mb",
                "video_path",
                "error",
            ]
        )

        for local_attn, sink in combos:
            tag = f"la{local_attn}_sk{sink}"
            run_dir = os.path.join(out_root, tag)
            os.makedirs(run_dir, exist_ok=True)
            out_mp4 = os.path.join(run_dir, "output.mp4")
            tmp_model: str | None = None
            generator = None
            t0 = time.perf_counter()
            peak_mb = ""

            try:
                tmp_model = build_model_dir_with_config(
                    args.model_path, local_attn, sink
                )
                generator = VideoGenerator.from_pretrained(
                    model_path=tmp_model,
                    num_gpus=1,
                    use_fsdp_inference=True,
                    dit_cpu_offload=True,
                    vae_cpu_offload=False,
                    text_encoder_cpu_offload=True,
                    pin_cpu_memory=True,
                )

                num_frames = args.num_frames
                actions = create_action_presets(num_frames, keyboard_dim=args.keyboard_dim)
                tensor = torch.zeros((num_frames, 23))
                tensor[:, args.keyboard_index] = 1.0
                actions["keyboard"] = tensor
                actions["mouse"] = torch.tensor([[0.0, 0.0]] * num_frames)

                result = generator.generate_video(
                    prompt="",
                    image_path=args.image_path,
                    mouse_cond=actions["mouse"].unsqueeze(0),
                    keyboard_cond=actions["keyboard"].unsqueeze(0),
                    grid_sizes=grid_sizes,
                    num_frames=num_frames,
                    height=args.height,
                    width=args.width,
                    num_inference_steps=args.num_inference_steps,
                    seed=args.seed,
                    output_path=out_mp4,
                    save_video=True,
                )
                elapsed = time.perf_counter() - t0
                peak = result.get("peak_memory_mb") if isinstance(result, dict) else None
                peak_mb = str(peak) if peak is not None else ""
                video_path = (
                    result.get("video_path") if isinstance(result, dict) else out_mp4
                )
                writer.writerow(
                    [local_attn, sink, f"{elapsed:.2f}", peak_mb, video_path, ""]
                )
                csvfile.flush()
                print(
                    f"[ok] {tag}  {elapsed:.1f}s  peak_mem={peak_mb}  -> {video_path}"
                )
                montage_paths.append(
                    str(video_path) if video_path else out_mp4
                )
                montage_labels.append(tag)
            except Exception as ex:  # noqa: BLE001
                err = str(ex)
                elapsed = time.perf_counter() - t0
                writer.writerow(
                    [local_attn, sink, f"{elapsed:.2f}", "", "", err]
                )
                csvfile.flush()
                print(f"[fail] {tag}: {err}")
            finally:
                if tmp_model and not args.keep_temp_dirs:
                    shutil.rmtree(tmp_model, ignore_errors=True)
                if generator is not None:
                    generator.shutdown()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    print(f"Summary CSV: {summary_path}")

    _try_ffmpeg_montage(
        args,
        out_root,
        montage_paths,
        montage_labels,
        len(combos),
    )


if __name__ == "__main__":
    main()
