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

# Ensure FastVideo package is importable when run as `python run_attn_full_context.py`
_FASTVIDEO_ROOT = Path(__file__).resolve().parent
if str(_FASTVIDEO_ROOT) not in sys.path:
    sys.path.insert(0, str(_FASTVIDEO_ROOT))


DEFAULT_MODEL = "/home/hal-kaiqin/models/skyreels_solaris"
DEFAULT_IMG = "/home/hal-kaiqin/models/image.png"
DEFAULT_OUT_ROOT = "video_samples_attn_full_context"
VAE_TIME_COMPRESSION_RATIO = 4
SOLARIS_KEYBOARD_DIM = 23
DEFAULT_SINK_SIZE = 0


def _ffprobe_bin() -> str:
    """Resolve ffprobe: PATH, or same directory as ffmpeg."""
    found = shutil.which("ffprobe")
    if found:
        return found
    ffm = shutil.which("ffmpeg")
    if ffm:
        cand = os.path.join(os.path.dirname(ffm), "ffprobe")
        if os.path.isfile(cand):
            return cand
    return "ffprobe"


def pixel_to_latent_frames(num_frames: int) -> int:
    """Convert pixel-frame count to latent-frame count for 4x temporal VAE."""
    if num_frames <= 0:
        raise ValueError(f"num_frames must be positive, got {num_frames}")
    if num_frames % VAE_TIME_COMPRESSION_RATIO != 1:
        raise ValueError(
            "Matrix-Game conditioning expects num_frames to be 4k+1; "
            f"got {num_frames}"
        )
    return (num_frames - 1) // VAE_TIME_COMPRESSION_RATIO + 1


def latent_to_pixel_frames(num_latent_frames: int) -> int:
    """Convert latent-frame count to pixel-frame count for 4x temporal VAE."""
    if num_latent_frames <= 0:
        raise ValueError(
            f"num_latent_frames must be positive, got {num_latent_frames}"
        )
    return (num_latent_frames - 1) * VAE_TIME_COMPRESSION_RATIO + 1


def build_model_dir_with_config(
    source_root: str,
    local_attn_size: int,
    sink_size: int,
) -> str:
    """Symlink-copy model tree; write a real transformer/config.json override."""
    source_root = os.path.abspath(source_root)
    tmp = tempfile.mkdtemp(prefix="matrixgame_fullctx_")
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
    """Decode first frame via imageio when ffprobe is unavailable."""
    import imageio.v3 as iio

    frame = iio.imread(path, index=0)
    h, w = int(frame.shape[0]), int(frame.shape[1])
    return w, h


def _verify_montage_output_shape(
    out_path: str,
    expected_w: int,
    expected_h: int,
) -> None:
    """Decode the written file's first frame and verify width/height."""
    import imageio.v3 as iio

    if not os.path.isfile(out_path) or os.path.getsize(out_path) == 0:
        raise RuntimeError(f"Montage output missing or empty: {out_path}")

    frame = iio.imread(out_path, index=0)
    h, w = int(frame.shape[0]), int(frame.shape[1])
    if w != expected_w or h != expected_h:
        raise RuntimeError(
            f"Montage size mismatch: decoded {w}x{h} != expected "
            f"{expected_w}x{expected_h}"
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
    """Concatenate videos into a cols x rows grid in row-major order."""
    n = len(video_paths)
    if n != cols * rows:
        raise ValueError(
            f"Need cols*rows ({cols}*{rows}={cols * rows}) videos, got {n}"
        )
    if labels is not None and len(labels) != n:
        raise ValueError(f"Need {n} labels, got {len(labels)}")
    for path in video_paths:
        if not os.path.isfile(path):
            raise FileNotFoundError(path)

    w0, h0 = _ffprobe_video_size(video_paths[0])
    tw, th = (w0 // 2) * 2, (h0 // 2) * 2
    if tw < 2 or th < 2:
        raise ValueError(f"Invalid tile size {tw}x{th} from first input")

    norm_parts = [
        f"[{i}:v]scale={tw}:{th}:force_original_aspect_ratio=decrease,"
        f"pad={tw}:{th}:(ow-iw)/2:(oh-ih)/2[ti{i}]"
        for i in range(n)
    ]
    layout = "|".join(
        f"{(i % cols) * tw}_{(i // cols) * th}"
        for i in range(n)
    )

    if labels:
        def _safe_label(text: str) -> str:
            if not all(c.isalnum() or c in "_=-" for c in text):
                raise ValueError(f"Unsafe montage label for ffmpeg: {text!r}")
            return text

        dt_parts: list[str] = []
        for i in range(n):
            label = _safe_label(labels[i])
            dt_parts.append(
                f"[ti{i}]drawtext=text='{label}':fontsize=22:fontcolor=white:"
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
    ffmpeg_exe = shutil.which("ffmpeg") or "ffmpeg"
    cmd: list[str] = [ffmpeg_exe, "-y"]
    for path in video_paths:
        cmd.extend(["-i", path])
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Run Solaris Matrix-Game inference with full-length local attention "
            "(no rolling KV cache)."
        )
    )
    p.add_argument(
        "--model-path",
        default=DEFAULT_MODEL,
        help="Directory containing skyreels_solaris (default: %(default)s)",
    )
    p.add_argument(
        "--output-root",
        default=DEFAULT_OUT_ROOT,
        help="Directory to store outputs and summary CSV",
    )
    p.add_argument("--image-path", default=DEFAULT_IMG)
    p.add_argument(
        "--latent-frame-sizes",
        type=int,
        nargs="+",
        default=[21, 42, 63, 84, 105, 126, 147],
        help=(
            "Latent-frame lengths to run independently; local_attn_size is set "
            "equal to each value (default: 21 42 63 84 105 126 147)"
        ),
    )
    p.add_argument("--height", type=int, default=352)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--num-inference-steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--keyboard-indices",
        type=int,
        nargs="+",
        default=[11, 12, 13, 14],
        help=(
            "Solaris keyboard channels to sweep; each run uses one active channel "
            "for all frames (default: 11 12 13 14)"
        ),
    )
    p.add_argument(
        "--keep-temp-dirs",
        action="store_true",
        help="Do not delete the symlink temp model dir after the run",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved config only; do not load models or generate",
    )
    p.add_argument(
        "--no-action-montage",
        action="store_true",
        help="Skip the per-length 2x2 action montage step",
    )
    p.add_argument(
        "--no-montage-labels",
        action="store_true",
        help="Build montages without drawtext labels",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    sink_size = DEFAULT_SINK_SIZE

    keyboard_indices = [int(v) for v in args.keyboard_indices]
    if not keyboard_indices:
        raise ValueError("keyboard_indices must not be empty")
    for keyboard_index in keyboard_indices:
        if keyboard_index < 0 or keyboard_index >= SOLARIS_KEYBOARD_DIM:
            raise ValueError(
                f"keyboard_index must be in [0, {SOLARIS_KEYBOARD_DIM - 1}], "
                f"got {keyboard_index}"
            )
    out_root = os.path.abspath(args.output_root)
    latent_sizes = [int(v) for v in args.latent_frame_sizes]
    if not latent_sizes:
        raise ValueError("latent_frame_sizes must not be empty")
    if not args.no_action_montage and len(keyboard_indices) != 4:
        raise ValueError(
            "Per-length 2x2 action montage expects exactly 4 keyboard indices; "
            f"got {len(keyboard_indices)}"
        )

    runs: list[tuple[int, int, int, int, str, str]] = []
    print(f"Planned runs: {len(latent_sizes) * len(keyboard_indices)}")
    for num_latent_frames in latent_sizes:
        num_frames = latent_to_pixel_frames(num_latent_frames)
        local_attn_size = pixel_to_latent_frames(num_frames)
        for keyboard_index in keyboard_indices:
            tag = f"la{local_attn_size}_sk{sink_size}_k{keyboard_index}"
            run_dir = os.path.join(out_root, tag)
            out_mp4 = os.path.join(run_dir, "output.mp4")
            runs.append(
                (
                    num_latent_frames,
                    num_frames,
                    local_attn_size,
                    keyboard_index,
                    run_dir,
                    out_mp4,
                )
            )
            print(
                "Resolved attention config: "
                f"num_latent_frames={num_latent_frames}, "
                f"num_frames={num_frames}, "
                f"local_attn_size={local_attn_size}, "
                f"sink_size={sink_size}, "
                f"keyboard_index={keyboard_index}"
            )
            if args.dry_run:
                print(f"  Output directory: {run_dir}")
                print(f"  Output video: {out_mp4}")
        if args.dry_run and not args.no_action_montage:
            montage_out = os.path.join(
                out_root, f"la{local_attn_size}_actions_2x2.mp4"
            )
            print(f"  Action montage: {montage_out}")

    if args.dry_run:
        return

    import torch
    from fastvideo import VideoGenerator

    os.makedirs(out_root, exist_ok=True)
    summary_path = os.path.join(
        out_root, f"full_context_summary_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    )
    successful_outputs: dict[tuple[int, int], str] = {}

    with open(summary_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            [
                "num_frames",
                "num_latent_frames",
                "local_attn_size",
                "sink_size",
                "keyboard_index",
                "seconds",
                "peak_memory_mb",
                "video_path",
                "error",
            ]
        )

        for (
            num_latent_frames,
            num_frames,
            local_attn_size,
            keyboard_index,
            run_dir,
            out_mp4,
        ) in runs:
            tag = f"la{local_attn_size}_sk{sink_size}_k{keyboard_index}"
            tmp_model: str | None = None
            generator = None
            t0 = time.perf_counter()

            os.makedirs(run_dir, exist_ok=True)

            try:
                tmp_model = build_model_dir_with_config(
                    args.model_path,
                    local_attn_size=local_attn_size,
                    sink_size=sink_size,
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

                grid_sizes = torch.tensor([num_latent_frames, 44, 80])
                keyboard = torch.zeros((num_frames, SOLARIS_KEYBOARD_DIM))
                keyboard[:, keyboard_index] = 1.0
                mouse = torch.zeros((num_frames, 2))

                result = generator.generate_video(
                    prompt="",
                    image_path=args.image_path,
                    mouse_cond=mouse.unsqueeze(0),
                    keyboard_cond=keyboard.unsqueeze(0),
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
                    [
                        num_frames,
                        num_latent_frames,
                        local_attn_size,
                        sink_size,
                        keyboard_index,
                        f"{elapsed:.2f}",
                        peak_mb,
                        video_path,
                        "",
                    ]
                )
                csvfile.flush()
                print(
                    f"[ok] {tag}  {elapsed:.1f}s  peak_mem={peak_mb}  -> {video_path}"
                )
                successful_outputs[(num_latent_frames, keyboard_index)] = str(video_path)
            except Exception as ex:  # noqa: BLE001
                err = str(ex)
                elapsed = time.perf_counter() - t0
                writer.writerow(
                    [
                        num_frames,
                        num_latent_frames,
                        local_attn_size,
                        sink_size,
                        keyboard_index,
                        f"{elapsed:.2f}",
                        "",
                        "",
                        err,
                    ]
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
    if args.no_action_montage:
        return

    for num_latent_frames in latent_sizes:
        local_attn_size = num_latent_frames
        montage_paths: list[str] = []
        montage_labels: list[str] = []
        missing: list[int] = []
        for keyboard_index in keyboard_indices:
            path = successful_outputs.get((num_latent_frames, keyboard_index))
            if path and os.path.isfile(path):
                montage_paths.append(path)
                montage_labels.append(f"k{keyboard_index}")
            else:
                missing.append(keyboard_index)
        montage_out = os.path.join(
            out_root, f"la{local_attn_size}_actions_2x2.mp4"
        )
        if missing:
            print(
                f"Montage skipped for la{local_attn_size}: missing outputs for "
                f"keyboard_index={missing}",
                file=sys.stderr,
            )
            continue
        try:
            ffmpeg_xstack_montage(
                montage_paths,
                montage_out,
                cols=2,
                rows=2,
                labels=None if args.no_montage_labels else montage_labels,
            )
            print(f"Action montage: {montage_out}")
        except FileNotFoundError as exc:
            print(
                f"Montage skipped for la{local_attn_size}: missing file or ffmpeg not in PATH: {exc}",
                file=sys.stderr,
            )
        except subprocess.CalledProcessError as exc:
            err = getattr(exc, "stderr", None) or getattr(exc, "stdout", None) or str(exc)
            print(
                f"Montage failed for la{local_attn_size}: {err}",
                file=sys.stderr,
            )


if __name__ == "__main__":
    main()
