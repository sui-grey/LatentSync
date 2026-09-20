# LatentSync HTTP server: load the model once, keep it warm, lip-sync over HTTP.
#
# The pipeline optimisations in this fork (reference cache, lazy face detector, batched restore,
# single encode) pay off most in a long-running process: after the first request for a reference
# video, every further request only pays for diffusion + a fraction of a second of glue.
#
#   pip install -r requirements_server.txt
#   python -m server --port 8092                      # LatentSync 1.5 (256px, ~8GB VRAM)
#   python -m server --version 1.6 --host 0.0.0.0     # 1.6 (512px, ~18GB), reachable from other machines
#
#   # 1) register a reference video once under a key
#   curl -F key=avatar -F video=@assets/demo1_video.mp4 http://127.0.0.1:8092/reference
#   # 2) lip-sync any number of audio clips against it
#   curl -F ref_key=avatar -F audio=@assets/demo1_audio.wav http://127.0.0.1:8092/lipsync/upload
#   #    -> {"video_url": "/download/1a2b3c4d.mp4", "duration": 11.84, "elapsed": 23.4, "timings": {...}}
#   curl -O http://127.0.0.1:8092/download/1a2b3c4d.mp4
#
# One job runs at a time (it owns the GPU); a second request gets HTTP 429 so the caller can retry or
# route elsewhere. There is no authentication: the default bind address is 127.0.0.1 on purpose.

import argparse
import os
import re
import shutil
import subprocess
import threading
import time
import uuid
from argparse import Namespace
from contextlib import asynccontextmanager
from typing import Optional

import torch
from accelerate.utils import set_seed
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from omegaconf import OmegaConf
from pydantic import BaseModel

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

VERSIONS = {
    "1.5": ("configs/unet/stage2.yaml", "checkpoints/latentsync_unet.pt"),
    "1.6": ("configs/unet/stage2_512.yaml", "checkpoints/latentsync_unet.pt"),
}

settings = Namespace()  # filled by main()
state = {}  # pipeline, config, dtype
gpu_lock = threading.Lock()  # one job at a time


class LipsyncRequest(BaseModel):
    audio_path: str  # a wav file readable by this server (e.g. written by a TTS server on the same machine)
    ref_key: str = "default"
    inference_steps: int = 20
    guidance_scale: float = 1.0
    seed: Optional[int] = None


class LipsyncResponse(BaseModel):
    video_path: str
    video_url: str
    duration: float
    elapsed: float
    timings: dict


def data_path(*parts):
    return os.path.join(settings.data_dir, *parts)


def reference_path(key: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", key) or key in (".", ".."):
        raise HTTPException(status_code=400, detail="key must match [A-Za-z0-9_.-]{1,64}")
    return data_path("references", key, "ref_video.mp4")


def media_duration(path: str) -> float:
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_entries", "format=duration", "-of", "csv=p=0", path],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return round(float(out.stdout.strip()), 2)
    except Exception:
        return 0.0


def prune_outputs():
    """Keep only the newest --keep_outputs results so a long-running server does not fill the disk."""
    if settings.keep_outputs <= 0:
        return
    out_dir = data_path("outputs")
    files = sorted(
        (os.path.join(out_dir, name) for name in os.listdir(out_dir) if name.endswith(".mp4")),
        key=os.path.getmtime,
    )
    for path in files[: -settings.keep_outputs]:
        try:
            os.remove(path)
        except OSError:
            pass


def run_job(audio_path: str, ref_key: str, inference_steps: int, guidance_scale: float, seed: Optional[int]):
    ref_path = reference_path(ref_key)
    if not os.path.exists(ref_path):
        raise HTTPException(status_code=404, detail=f"Reference '{ref_key}' not found; POST /reference first")
    if not os.path.exists(audio_path):
        raise HTTPException(status_code=404, detail=f"Audio not found: {audio_path}")
    if "pipeline" not in state:
        raise HTTPException(status_code=503, detail="Pipeline not loaded yet")
    if not gpu_lock.acquire(blocking=False):
        raise HTTPException(status_code=429, detail="Server busy, try again later")

    try:
        pipeline, config, dtype = state["pipeline"], state["config"], state["dtype"]
        job_id = uuid.uuid4().hex[:8]
        out_path = data_path("outputs", f"{job_id}.mp4")
        if seed is not None:
            set_seed(seed)
        start = time.perf_counter()
        pipeline(
            video_path=ref_path,
            audio_path=audio_path,
            video_out_path=out_path,
            num_frames=config.data.num_frames,
            num_inference_steps=inference_steps,
            guidance_scale=guidance_scale,
            weight_dtype=dtype,
            width=config.data.resolution,
            height=config.data.resolution,
            mask_image_path=config.data.mask_image_path,
            temp_dir=data_path("temp"),
            ref_cache=not settings.no_ref_cache,
            ref_cache_dir=data_path("ref_cache"),
            legacy_encode=settings.legacy_encode,
            restore_batch_size=settings.restore_batch_size,
        )
        elapsed = round(time.perf_counter() - start, 2)
        timings = {k: round(v, 2) for k, v in getattr(pipeline, "last_timings", {}).items()}
        state["last_timings"] = timings
        state["jobs_done"] = state.get("jobs_done", 0) + 1
        prune_outputs()
        return LipsyncResponse(
            video_path=out_path,
            video_url=f"/download/{job_id}.mp4",
            duration=media_duration(out_path),
            elapsed=elapsed,
            timings=timings,
        )
    finally:
        gpu_lock.release()


@asynccontextmanager
async def lifespan(app: FastAPI):
    from scripts.inference import build_pipeline

    for sub in ("references", "outputs", "uploads", "temp"):
        os.makedirs(data_path(sub), exist_ok=True)
    config = OmegaConf.load(settings.unet_config_path)
    start = time.perf_counter()
    state["pipeline"], state["dtype"] = build_pipeline(config, settings)
    state["config"] = config
    print(f"LatentSync loaded in {time.perf_counter() - start:.1f}s — ready on http://{settings.host}:{settings.port}")
    yield
    state.clear()


app = FastAPI(title="LatentSync server", lifespan=lifespan)


# Plain (non-async) handlers: FastAPI runs them in a worker thread, so /health keeps answering
# while a job holds the GPU.


@app.get("/health")
def health():
    return {
        "status": "ok",
        "busy": gpu_lock.locked(),
        "model": f"v{settings.version}",
        "pipeline_loaded": "pipeline" in state,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "jobs_done": state.get("jobs_done", 0),
        "last_timings": state.get("last_timings"),
    }


@app.post("/reference")
def upload_reference(key: str = Form(...), video: UploadFile = File(...)):
    path = reference_path(key)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        shutil.copyfileobj(video.file, f)
    return {"key": key, "video": path}  # a new file changes the cache key, so stale cache entries are never used


@app.get("/reference")
def list_references():
    root = data_path("references")
    return {"references": sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))}


@app.post("/lipsync", response_model=LipsyncResponse)
def lipsync(req: LipsyncRequest):
    """Audio already on this machine (path). Use /lipsync/upload from another machine."""
    return run_job(req.audio_path, req.ref_key, req.inference_steps, req.guidance_scale, req.seed)


@app.post("/lipsync/upload", response_model=LipsyncResponse)
def lipsync_upload(
    audio: UploadFile = File(...),
    ref_key: str = Form("default"),
    inference_steps: int = Form(20),
    guidance_scale: float = Form(1.0),
    seed: Optional[int] = Form(None),
):
    if gpu_lock.locked():  # refuse before reading the upload
        raise HTTPException(status_code=429, detail="Server busy, try again later")
    audio_path = data_path("uploads", f"{uuid.uuid4().hex[:8]}.wav")
    with open(audio_path, "wb") as f:
        shutil.copyfileobj(audio.file, f)
    try:
        return run_job(audio_path, ref_key, inference_steps, guidance_scale, seed)
    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)


@app.get("/download/{filename}")
def download(filename: str):
    path = data_path("outputs", os.path.basename(filename))
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path, media_type="video/mp4", filename=os.path.basename(filename))


def main():
    import uvicorn

    parser = argparse.ArgumentParser(description="LatentSync HTTP server")
    parser.add_argument("--version", choices=sorted(VERSIONS), default="1.5", help="1.5 = 256px (~8GB), 1.6 = 512px (~18GB)")
    parser.add_argument("--unet_config_path", type=str, default=None, help="overrides --version")
    parser.add_argument("--inference_ckpt_path", type=str, default=None, help="overrides --version")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="0.0.0.0 to accept other machines (no auth!)")
    parser.add_argument("--port", type=int, default=8092)
    parser.add_argument(
        "--data_dir",
        type=str,
        default=os.environ.get("LATENTSYNC_SERVER_DATA", os.path.join(REPO_ROOT, "server_data")),
        help="references/, outputs/, uploads/, ref_cache/ live here (env LATENTSYNC_SERVER_DATA)",
    )
    parser.add_argument("--keep_outputs", type=int, default=50, help="keep the newest N result videos (0 = keep all)")
    parser.add_argument("--no_deepcache", action="store_true")
    parser.add_argument("--no_ref_cache", action="store_true", help="stock behaviour, for comparison")
    parser.add_argument("--legacy_encode", action="store_true", help="stock behaviour, for comparison")
    parser.add_argument("--restore_batch_size", type=int, default=16, help="1 = stock per-frame loop")
    parser.parse_args(namespace=settings)

    default_config, default_ckpt = VERSIONS[settings.version]
    settings.unet_config_path = settings.unet_config_path or default_config
    settings.inference_ckpt_path = settings.inference_ckpt_path or default_ckpt
    settings.enable_deepcache = not settings.no_deepcache
    settings.data_dir = os.path.abspath(settings.data_dir)

    os.chdir(REPO_ROOT)  # configs/, checkpoints/ and the mask image are addressed relative to the repo root
    print(f"LatentSync server v{settings.version} | data: {settings.data_dir}")
    uvicorn.run(app, host=settings.host, port=settings.port)


if __name__ == "__main__":
    main()
