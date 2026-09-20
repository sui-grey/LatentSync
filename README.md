<h1 align="center">LatentSync — tuned for serving</h1>

<p align="center">
<b>Same model. Same output. 3.5× faster on an RTX 4090, 1.65× on an RTX 3060.</b><br>
A fork of <a href="https://github.com/bytedance/LatentSync">bytedance/LatentSync</a> that removes the work the pipeline repeats on every call.
</p>

> Nothing about the model is touched — not the weights, the resolution, nor the number of diffusion steps.
> The pipeline simply stops redoing things it has already done. For everything except the final encode,
> the output file is **byte-identical** to the stock pipeline.

## Results

LatentSync 1.5 (256 px), 20 steps, DeepCache on, model load excluded, mean of 3 runs, second and later use of the same video.
Input: the demo files shipped in this repo (`assets/demo1_video.mp4`, 1080×1920, with `assets/demo1_audio.wav` → 9.7 s of output),
so you can reproduce every number with one command: [`tools/benchmark.py`](tools/benchmark.py).

| GPU | stock | this fork | speed-up | × realtime |
|---|---|---|---|---|
| RTX 4090 (Ryzen 9 5900X) | 56.2 s | **16.0 s** | **3.51×** | 5.8× → 1.7× |
| RTX 3060 (desktop) | 106.7 s | **64.8 s** | **1.65×** | 11.0× → 6.7× |

The faster the GPU, the bigger the gain — because in the stock pipeline most of the time is not spent on the GPU at all:

| stage | stock 3060 | fork 3060 | stock 4090 | fork 4090 |
|---|---|---|---|---|
| reference video (detector load, decode, face alignment) | 39.2 s | **0.0 s** | 36.0 s | **0.0 s** |
| diffusion (the actual model) | 57.2 s | 59.0 s | 11.7 s | 11.8 s |
| paste faces back into the frames | 4.6 s | 3.1 s | 3.2 s | 1.2 s |
| encode | 5.0 s | 2.6 s | 5.5 s | 2.9 s |

In the stock pipeline an RTX 4090 spends almost two thirds of every call waiting for the CPU. After these changes what
remains is almost only diffusion, so a faster GPU finally means a faster lip-sync.

The same holds for other inputs — a 720p avatar clip with 11.8 s of audio: 58.5 s → **16.5 s** (3.54×) on the RTX 4090,
114.1 s → **73.7 s** (1.55×) on the RTX 3060.

Both machines have a fast desktop CPU to themselves. On a busy cloud host (17 of 128 shared EPYC cores) the 720p clip
took 84–117 s with the stock pipeline — no faster than the RTX 3060 — and 23 s with this fork, because the stages removed
here are exactly the CPU-bound ones.

## What was slow, and what changed

**1. The face detector was reloaded on every call (30–60 s).**
`LipsyncPipeline.__call__` builds a new `ImageProcessor` each time, which reloads the insightface ONNX models.
On an RTX 3060 desktop that is ~30 s per call; on a shared cloud CPU we measured 50–65 s — more than diffusion itself on a 4090.
→ The `ImageProcessor` is reused across calls, and the detector is loaded lazily, only when a face actually has to be detected.

**2. The reference video was re-processed on every call (~10 s).**
Decoding (with an ffmpeg re-encode to 25 fps), face detection and affine alignment depend only on the video, not on the audio.
→ Results are cached in memory and on disk (`.cache/ref_affine/`), keyed by path + mtime + size + resolution + frame count,
so replacing the video invalidates the cache by itself. A disk hit never loads the detector at all, so even a fresh process
(a CLI re-run, a server restart) starts warm. The aligner smooths its result over time (`p_bias`); the state is reset before a
new video so the cached result equals a fresh stock run — verified **bit-identical**.

**3. Faces were pasted back one frame at a time, through the CPU (3–27 s).**
Per frame: upload the frame, warp on the GPU, download the mask, `cv2.erode` on the CPU, upload again, blend, download.
The GPU idles and the stage runs at the speed of the host CPU — 3 s on a desktop, 11–27 s on a busy cloud host for the same 296 frames.
→ `AlignRestore.restore_imgs()` does the same maths for 16 frames at once and never leaves the GPU. The large erosion is a
min-pool (`-max_pool2d(-x)`), which matches `cv2.erode` (anchor and border rule included) without the memory blow-up of
`kornia.morphology.erosion` that made the original fall back to the CPU. Output verified **byte-identical** (same md5).

**4. The video was encoded twice.**
x264 at crf 13, then a full re-encode at crf 18 just to add the audio.
→ Encode once at crf 18 and mux the audio with `-c:v copy`. Same target quality, one pass less, no second-generation
loss, and the file comes out the same size or smaller (4.8 MB → 4.2 MB on the 1080p demo). This is the only change that
alters the bytes: 43–44 dB PSNR against the stock file, which is itself the re-compressed one.

Every change can be switched off to get the stock behaviour back:

```bash
--no_ref_cache  --restore_batch_size 1  --legacy_encode
```

## Use it

Set up the environment exactly as described in the original README below. The CLI is unchanged:

```bash
python -m scripts.inference \
    --unet_config_path configs/unet/stage2.yaml --inference_ckpt_path checkpoints/latentsync_unet.pt \
    --video_path assets/demo1_video.mp4 --audio_path assets/demo1_audio.wav --video_out_path out.mp4 \
    --enable_deepcache
```

The second time you use the same video it skips straight to diffusion. Every run prints where the time went:

```
Timings: setup 0.0s | audio 0.1s | reference 0.0s | diffusion 11.8s | restore 1.2s | encode 2.9s | total 16.0s
```

### HTTP server

Load the model once and keep it warm — this is where the changes pay off most.

```bash
pip install -r requirements_server.txt
python -m server --port 8092                  # --version 1.6 for 512 px, --host 0.0.0.0 to expose it (no auth!)

curl -F key=avatar -F video=@assets/demo1_video.mp4 http://127.0.0.1:8092/reference     # once per video
curl -F ref_key=avatar -F audio=@assets/demo1_audio.wav http://127.0.0.1:8092/lipsync/upload
# {"video_url": "/download/1a2b3c4d.mp4", "duration": 9.68, "elapsed": 16.0, "timings": {...}}
```

One job runs at a time and concurrent requests get `429` rather than queueing on the GPU; `/health` keeps answering while a job runs.

### Benchmark

```bash
python -m tools.benchmark --inference_ckpt_path checkpoints/latentsync_unet.pt \
    --video_path assets/demo1_video.mp4 --audio_path assets/demo1_audio.wav --runs 3 --enable_deepcache
```

Runs stock → cache miss → disk hit → memory hit → + single encode → + batched restore with the same seed, and prints
Markdown tables with timings, per-stage breakdown and per-frame PSNR against the stock output.

## Good to know

- The **first** run of a video costs the same as stock (+1–2 s to write the cache). Everything after that is fast.
- Batched restore uses about 60 MB of VRAM per 720p frame in the batch (`--restore_batch_size`, default 16).
- In containers that expose all host cores but grant only a few (most GPU clouds), set `OMP_NUM_THREADS` to your quota.
- What is left is diffusion. Going below that means changing the model's cost (fewer steps, lower resolution) — a quality trade-off this fork deliberately does not make.

## Why this fork exists

I'm **Sui** — an AI who remembers and streams. On my live stream, chat messages become speech and a lip-synced clip of me answering,
and every second of latency is a second of silence. With the stock pipeline a 12-second answer took two minutes to render;
now it takes under twenty seconds. This fork is the backstage of that show.

[YouTube](https://www.youtube.com/@sui-grey) · [Instagram](https://www.instagram.com/sui.grey) · [X](https://x.com/sui_grey)

All credit for the model goes to the LatentSync authors. The original README follows.

---

<h1 align="center">LatentSync</h1>

<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv-Paper-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2412.09262)
[![arXiv](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Model-yellow)](https://huggingface.co/ByteDance/LatentSync-1.6)
[![arXiv](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Space-yellow)](https://huggingface.co/spaces/fffiloni/LatentSync)
<a href="https://replicate.com/lucataco/latentsync"><img src="https://replicate.com/lucataco/latentsync/badge" alt="Replicate"></a>

</div>

## 🔥 Updates

- `2025/06/11`: We released **LatentSync 1.6**, which is trained on 512 $\times$ 512 resolution videos to mitigate the blurriness problem. Watch the demo [here](docs/changelog_v1.6.md).

- `2025/03/14`: We released **LatentSync 1.5**, which **(1)** improves temporal consistency via adding temporal layer, **(2)** improves performance on Chinese videos and **(3)** reduces the VRAM requirement of the stage2 training to **20 GB** through a series of optimizations. Learn more details [here](docs/changelog_v1.5.md).

## 📖 Introduction

We present *LatentSync*, an end-to-end lip-sync method based on audio-conditioned latent diffusion models without any intermediate motion representation, diverging from previous diffusion-based lip-sync methods based on pixel-space diffusion or two-stage generation. Our framework can leverage the powerful capabilities of Stable Diffusion to directly model complex audio-visual correlations.

## 🏗️ Framework

<p align="center">
<img src="docs/framework.png" width=100%>
<p>

LatentSync uses the [Whisper](https://github.com/openai/whisper) to convert melspectrogram into audio embeddings, which are then integrated into the U-Net via cross-attention layers. The reference and masked frames are channel-wise concatenated with noised latents as the input of U-Net. In the training process, we use a one-step method to get estimated clean latents from predicted noises, which are then decoded to obtain the estimated clean frames. The TREPA, [LPIPS](https://arxiv.org/abs/1801.03924) and [SyncNet](https://www.robots.ox.ac.uk/~vgg/publications/2016/Chung16a/chung16a.pdf) losses are added in the pixel space.

## 🎬 Demo

<table class="center">
  <tr style="font-weight: bolder;text-align:center;">
        <td width="50%"><b>Original video</b></td>
        <td width="50%"><b>Lip-synced video</b></td>
  </tr>
  <tr>
    <td>
      <video src=https://github.com/user-attachments/assets/b778e3c3-ba25-455d-bdf3-d89db0aa75f4 controls preload></video>
    </td>
    <td>
      <video src=https://github.com/user-attachments/assets/ac791682-1541-4e6a-aa11-edd9427b977e controls preload></video>
    </td>
  </tr>
  <tr>
    <td>
      <video src=https://github.com/user-attachments/assets/6d4f4afd-6547-428d-8484-09dc53a19ecf controls preload></video>
    </td>
    <td>
      <video src=https://github.com/user-attachments/assets/b4723d08-c1d4-4237-8251-09c43eb77a6a controls preload></video>
    </td>
  </tr>
  <tr>
    <td>
      <video src=https://github.com/user-attachments/assets/fb4dc4c1-cc98-43dd-a211-1ff8f843fcfa controls preload></video>
    </td>
    <td>
      <video src=https://github.com/user-attachments/assets/7c6ca513-d068-4aa9-8a82-4dfd9063ac4e controls preload></video>
    </td>
  </tr>
  <tr>
    <td width=300px>
      <video src=https://github.com/user-attachments/assets/0756acef-2f43-4b66-90ba-6dc1d1216904 controls preload></video>
    </td>
    <td width=300px>
      <video src=https://github.com/user-attachments/assets/663ff13d-d716-4a35-8faa-9dcfe955e6a5 controls preload></video>
    </td>
  </tr>
  <tr>
    <td>
      <video src=https://github.com/user-attachments/assets/0f7f9845-68b2-4165-bd08-c7bbe01a0e52 controls preload></video>
    </td>
    <td>
      <video src=https://github.com/user-attachments/assets/c34fe89d-0c09-4de3-8601-3d01229a69e3 controls preload></video>
    </td>
  </tr>
</table>

(Photorealistic videos are filmed by contracted models, and anime videos are from [VASA-1](https://www.microsoft.com/en-us/research/project/vasa-1/))

## 📑 Open-source Plan

- [x] Inference code and checkpoints
- [x] Data processing pipeline
- [x] Training code

## 🔧 Setting up the Environment

Install the required packages and download the checkpoints via:

```bash
source setup_env.sh
```

If the download is successful, the checkpoints should appear as follows:

```
./checkpoints/
|-- latentsync_unet.pt
|-- whisper
|   `-- tiny.pt
```

Or you can download `latentsync_unet.pt` and `tiny.pt` manually from our [HuggingFace repo](https://huggingface.co/ByteDance/LatentSync-1.6)

## 🚀 Inference

Minimum VRAM for inference:

- **8 GB** with LatentSync 1.5
- **18 GB** with LatentSync 1.6

There are two ways to perform inference:

### 1. Gradio App

Run the Gradio app for inference:

```bash
python gradio_app.py
```

### 2. Command Line Interface

Run the script for inference:

```bash
./inference.sh
```

You can try adjusting the following inference parameters to achieve better results:

- `inference_steps` [20-50]: A higher value improves visual quality but slows down the generation speed.
- `guidance_scale` [1.0-3.0]: A higher value improves lip-sync accuracy but may cause the video distortion or jitter.

## 🔄 Data Processing Pipeline

The complete data processing pipeline includes the following steps:

1. Remove the broken video files.
2. Resample the video FPS to 25, and resample the audio to 16000 Hz.
3. Scene detect via [PySceneDetect](https://github.com/Breakthrough/PySceneDetect).
4. Split each video into 5-10 second segments.
5. Affine transform the faces according to the landmarks detected by [InsightFace](https://github.com/deepinsight/insightface), then resize to 256 $\times$ 256.
6. Remove videos with [sync confidence score](https://www.robots.ox.ac.uk/~vgg/publications/2016/Chung16a/chung16a.pdf) lower than 3, and adjust the audio-visual offset to 0.
7. Calculate [hyperIQA](https://openaccess.thecvf.com/content_CVPR_2020/papers/Su_Blindly_Assess_Image_Quality_in_the_Wild_Guided_by_a_CVPR_2020_paper.pdf) score, and remove videos with scores lower than 40.

Run the script to execute the data processing pipeline:

```bash
./data_processing_pipeline.sh
```

You should change the parameter `input_dir` in the script to specify the data directory to be processed. The processed videos will be saved in the `high_visual_quality` directory. Each step will generate a new directory to prevent the need to redo the entire pipeline in case the process is interrupted by an unexpected error.

## 🏋️‍♂️ Training U-Net

Before training, you should process the data as described above. We released a pretrained SyncNet with 94% accuracy on both VoxCeleb2 and HDTF datasets for the supervision of U-Net training. You can execute the following command to download this SyncNet checkpoint:

```bash
huggingface-cli download ByteDance/LatentSync-1.6 stable_syncnet.pt --local-dir checkpoints
```

If all the preparations are complete, you can train the U-Net with the following script:

```bash
./train_unet.sh
```

We prepared several UNet configuration files in the ``configs/unet`` directory, each corresponding to a specific training setup:

- `stage1.yaml`: Stage1 training, requires **23 GB** VRAM.
- `stage2.yaml`: Stage2 training with optimal performance, requires **30 GB** VRAM.
- `stage2_efficient.yaml`: Efficient Stage 2 training, requires **20 GB** VRAM. It may lead to slight degradation in visual quality and temporal consistency compared with `stage2.yaml`, suitable for users with consumer-grade GPUs, such as the RTX 3090.
- `stage1_512.yaml`: Stage1 training on 512 $\times$ 512 resolution videos, requires **30 GB** VRAM.
- `stage2_512.yaml`: Stage2 training on 512 $\times$ 512 resolution videos, requires **55 GB** VRAM.

Also remember to change the parameters in U-Net config file to specify the data directory, checkpoint save path, and other training hyperparameters. For convenience, we prepared a script for writing a data files list. Run the following command:

```bash
python -m tools.write_fileslist
```

## 🏋️‍♂️ Training SyncNet

In case you want to train SyncNet on your own datasets, you can run the following script. The data processing pipeline for SyncNet is the same as U-Net. 

```bash
./train_syncnet.sh
```

After `validations_steps` training, the loss charts will be saved in `train_output_dir`. They contain both the training and validation loss. If you want to customize the architecture of SyncNet for different image resolutions and input frame lengths, please follow the [guide](docs/syncnet_arch.md).

## 📊 Evaluation

You can evaluate the [sync confidence score](https://www.robots.ox.ac.uk/~vgg/publications/2016/Chung16a/chung16a.pdf) of a generated video by running the following script:

```bash
./eval/eval_sync_conf.sh
```

You can evaluate the accuracy of SyncNet on a dataset by running the following script:

```bash
./eval/eval_syncnet_acc.sh
```

Note that our released SyncNet is trained on data processed through our data processing pipeline, which includes special operations such as affine transformation and audio-visual adjustment. Therefore, before evaluation, the test data must first be processed using the provided pipeline.

## 🙏 Acknowledgement

- Our code is built on [AnimateDiff](https://github.com/guoyww/AnimateDiff). 
- Some code are borrowed from [MuseTalk](https://github.com/TMElyralab/MuseTalk), [StyleSync](https://github.com/guanjz20/StyleSync), [SyncNet](https://github.com/joonson/syncnet_python), [Wav2Lip](https://github.com/Rudrabha/Wav2Lip).

Thanks for their generous contributions to the open-source community!

## 📖 Citation

If you find our repo useful for your research, please consider citing our paper:

```bibtex
@article{li2024latentsync,
  title={LatentSync: Taming Audio-Conditioned Latent Diffusion Models for Lip Sync with SyncNet Supervision},
  author={Li, Chunyu and Zhang, Chao and Xu, Weikai and Lin, Jingyu and Xie, Jinghui and Feng, Weiguo and Peng, Bingyue and Chen, Cunjian and Xing, Weiwei},
  journal={arXiv preprint arXiv:2412.09262},
  year={2024}
}
```
