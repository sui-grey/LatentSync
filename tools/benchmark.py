# Benchmark the stock pipeline against the optimisations in this fork, on your own GPU.
#
# The model is loaded once; every mode then runs the same video + audio with the same seed,
# so the numbers isolate the pipeline work (not model loading) and the outputs are comparable.
# Besides timing, the output of every mode is compared frame by frame with the stock output
# (PSNR), to show that the optimisations do not change the result.
#
# Usage (from the repo root, same arguments as scripts.inference):
#   python -m tools.benchmark \
#       --unet_config_path configs/unet/stage2.yaml \
#       --inference_ckpt_path checkpoints/latentsync_unet.pt \
#       --video_path assets/demo1_video.mp4 --audio_path assets/demo1_audio.wav \
#       --runs 2 --enable_deepcache
#
# Prints a Markdown table that can be pasted into an issue / README.

import argparse
import os
import shutil
import time

import cv2
import numpy as np
import torch
from accelerate.utils import set_seed
from omegaconf import OmegaConf

from scripts.inference import build_pipeline


def clear_reference_state(pipeline):
    """Forget everything a fresh process would not have (simulates a new CLI run / server restart)."""
    pipeline.__dict__.pop("_ref_frames_cache", None)
    pipeline.__dict__.pop("_ref_affine_cache", None)
    pipeline.image_processor = None


def run_once(pipeline, config, dtype, args, out_path, **overrides):
    set_seed(args.seed)
    torch.cuda.synchronize()
    start = time.perf_counter()
    pipeline(
        video_path=args.video_path,
        audio_path=args.audio_path,
        video_out_path=out_path,
        num_frames=config.data.num_frames,
        num_inference_steps=args.inference_steps,
        guidance_scale=args.guidance_scale,
        weight_dtype=dtype,
        width=config.data.resolution,
        height=config.data.resolution,
        mask_image_path=config.data.mask_image_path,
        temp_dir=args.temp_dir,
        **overrides,
    )
    torch.cuda.synchronize()
    return time.perf_counter() - start


def read_frames(path):
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    return frames


def psnr_against(reference_path, other_path):
    """Returns (mean PSNR, min PSNR, frame count). inf means bit-identical decoded frames."""
    ref, other = read_frames(reference_path), read_frames(other_path)
    n = min(len(ref), len(other))
    values = []
    for a, b in zip(ref[:n], other[:n]):
        mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
        values.append(float("inf") if mse == 0 else 10 * np.log10(255.0**2 / mse))
    return float(np.mean(values)), float(np.min(values)), n


def video_seconds(path):
    cap = cv2.VideoCapture(path)
    frames, fps = cap.get(cv2.CAP_PROP_FRAME_COUNT), cap.get(cv2.CAP_PROP_FPS) or 25
    cap.release()
    return frames / fps


def fmt_psnr(value):
    return "identical" if value == float("inf") else f"{value:.1f} dB"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--unet_config_path", type=str, default="configs/unet/stage2.yaml")
    parser.add_argument("--inference_ckpt_path", type=str, required=True)
    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument("--audio_path", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="benchmark_out")
    parser.add_argument("--runs", type=int, default=2, help="timed runs per mode")
    parser.add_argument("--inference_steps", type=int, default=20)
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--temp_dir", type=str, default="temp")
    parser.add_argument("--seed", type=int, default=1247)
    parser.add_argument("--enable_deepcache", action="store_true")
    parser.add_argument("--only", type=str, default="", help="comma separated mode indices to run besides stock, e.g. 3,4")
    args = parser.parse_args()

    config = OmegaConf.load(args.unet_config_path)
    pipeline, dtype = build_pipeline(config, args)

    os.makedirs(args.out_dir, exist_ok=True)
    cache_dir = os.path.join(args.out_dir, "ref_cache")

    # name, pipeline kwargs, what to reset before EVERY run of this mode
    def reset_cold_no_disk():
        clear_reference_state(pipeline)
        shutil.rmtree(cache_dir, ignore_errors=True)

    cached = dict(ref_cache=True, ref_cache_dir=cache_dir)
    legacy = dict(legacy_encode=True, restore_batch_size=1)
    modes = [
        ("stock", dict(ref_cache=False, **legacy), lambda: clear_reference_state(pipeline)),
        ("ref cache: miss (first ever run)", dict(**legacy, **cached), reset_cold_no_disk),
        ("ref cache: disk hit (new process)", dict(**legacy, **cached), lambda: clear_reference_state(pipeline)),
        ("ref cache: memory hit (serving)", dict(**legacy, **cached), lambda: None),
        ("+ single encode", dict(legacy_encode=False, restore_batch_size=1, **cached), lambda: None),
        ("+ batched restore (all on)", dict(legacy_encode=False, restore_batch_size=16, **cached), lambda: None),
    ]
    if args.only:
        keep = {0} | {int(i) for i in args.only.split(",")}  # stock is always needed as the reference
        modes = [m for i, m in enumerate(modes) if i in keep]

    # Warm-up: CUDA kernels / cudnn autotune, so the first timed mode is not penalised. It also fills the
    # reference cache, so the "hit" modes are real hits even when earlier modes are skipped with --only.
    print("\n=== warm-up ===")
    clear_reference_state(pipeline)
    run_once(pipeline, config, dtype, args, os.path.join(args.out_dir, "warmup.mp4"), **legacy, **cached)

    results, stage_rows = [], []
    for index, (name, overrides, reset) in enumerate(modes):
        out_path = os.path.join(args.out_dir, f"mode{index}.mp4")
        times = []
        for run in range(args.runs):
            print(f"\n=== {name} — run {run + 1}/{args.runs} ===")
            reset()
            times.append(run_once(pipeline, config, dtype, args, out_path, **overrides))
        results.append((name, times, out_path))
        stage_rows.append((name, dict(getattr(pipeline, "last_timings", {}))))

    stock_path = results[0][2]
    stock_mean = float(np.mean(results[0][1]))
    audio_s = video_seconds(stock_path)

    print("\n\n## LatentSync pipeline benchmark\n")
    print(
        f"- GPU: {torch.cuda.get_device_name(0)} | torch {torch.__version__} | "
        f"resolution {config.data.resolution} | steps {args.inference_steps} | "
        f"deepcache {'on' if args.enable_deepcache else 'off'}"
    )
    print(f"- output: {audio_s:.1f}s of video, {args.runs} runs per mode, model load excluded\n")
    print("| mode | mean | best | vs stock | x realtime | PSNR vs stock (mean / min) | size |")
    print("|---|---|---|---|---|---|---|")
    for name, times, out_path in results:
        mean, best = float(np.mean(times)), float(np.min(times))
        mean_psnr, min_psnr, _ = psnr_against(stock_path, out_path)
        print(
            f"| {name} | {mean:.1f}s | {best:.1f}s | {stock_mean / mean:.2f}x | {mean / audio_s:.1f}x | "
            f"{fmt_psnr(mean_psnr)} / {fmt_psnr(min_psnr)} | {os.path.getsize(out_path) / 1e6:.1f} MB |"
        )

    stages = ["audio", "reference", "diffusion", "restore", "encode", "total"]
    print("\nWhere the time goes (last run of each mode):\n")
    print("| mode | " + " | ".join(stages) + " |")
    print("|---|" + "---|" * len(stages))
    for name, timings in stage_rows:
        print(f"| {name} | " + " | ".join(f"{timings.get(stage, 0):.1f}s" for stage in stages) + " |")


if __name__ == "__main__":
    main()
