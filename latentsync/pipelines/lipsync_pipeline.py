# Adapted from https://github.com/guoyww/AnimateDiff/blob/main/animatediff/pipelines/pipeline_animation.py

import hashlib
import inspect
import math
import os
import shutil
import time
from typing import Callable, List, Optional, Union
import subprocess

import numpy as np
import torch
import torchvision
from torchvision import transforms

from packaging import version

from diffusers.configuration_utils import FrozenDict
from diffusers.models import AutoencoderKL
from diffusers.pipelines import DiffusionPipeline
from diffusers.schedulers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
)
from diffusers.utils import deprecate, logging

from einops import rearrange
import cv2

from ..models.unet import UNet3DConditionModel
from ..utils.util import read_video, read_audio, write_video, check_ffmpeg_installed
from ..utils.image_processor import ImageProcessor, load_fixed_mask
from ..whisper.audio2feature import Audio2Feature
import tqdm
import soundfile as sf

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class LipsyncPipeline(DiffusionPipeline):
    _optional_components = []

    def __init__(
        self,
        vae: AutoencoderKL,
        audio_encoder: Audio2Feature,
        unet: UNet3DConditionModel,
        scheduler: Union[
            DDIMScheduler,
            PNDMScheduler,
            LMSDiscreteScheduler,
            EulerDiscreteScheduler,
            EulerAncestralDiscreteScheduler,
            DPMSolverMultistepScheduler,
        ],
    ):
        super().__init__()

        if hasattr(scheduler.config, "steps_offset") and scheduler.config.steps_offset != 1:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} is outdated. `steps_offset`"
                f" should be set to 1 instead of {scheduler.config.steps_offset}. Please make sure "
                "to update the config accordingly as leaving `steps_offset` might led to incorrect results"
                " in future versions. If you have downloaded this checkpoint from the Hugging Face Hub,"
                " it would be very nice if you could open a Pull request for the `scheduler/scheduler_config.json`"
                " file"
            )
            deprecate("steps_offset!=1", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["steps_offset"] = 1
            scheduler._internal_dict = FrozenDict(new_config)

        if hasattr(scheduler.config, "clip_sample") and scheduler.config.clip_sample is True:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} has not set the configuration `clip_sample`."
                " `clip_sample` should be set to False in the configuration file. Please make sure to update the"
                " config accordingly as not setting `clip_sample` in the config might lead to incorrect results in"
                " future versions. If you have downloaded this checkpoint from the Hugging Face Hub, it would be very"
                " nice if you could open a Pull request for the `scheduler/scheduler_config.json` file"
            )
            deprecate("clip_sample not set", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["clip_sample"] = False
            scheduler._internal_dict = FrozenDict(new_config)

        is_unet_version_less_0_9_0 = hasattr(unet.config, "_diffusers_version") and version.parse(
            version.parse(unet.config._diffusers_version).base_version
        ) < version.parse("0.9.0.dev0")
        is_unet_sample_size_less_64 = hasattr(unet.config, "sample_size") and unet.config.sample_size < 64
        if is_unet_version_less_0_9_0 and is_unet_sample_size_less_64:
            deprecation_message = (
                "The configuration file of the unet has set the default `sample_size` to smaller than"
                " 64 which seems highly unlikely. If your checkpoint is a fine-tuned version of any of the"
                " following: \n- CompVis/stable-diffusion-v1-4 \n- CompVis/stable-diffusion-v1-3 \n-"
                " CompVis/stable-diffusion-v1-2 \n- CompVis/stable-diffusion-v1-1 \n- runwayml/stable-diffusion-v1-5"
                " \n- runwayml/stable-diffusion-inpainting \n you should change 'sample_size' to 64 in the"
                " configuration file. Please make sure to update the config accordingly as leaving `sample_size=32`"
                " in the config might lead to incorrect results in future versions. If you have downloaded this"
                " checkpoint from the Hugging Face Hub, it would be very nice if you could open a Pull request for"
                " the `unet/config.json` file"
            )
            deprecate("sample_size<64", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(unet.config)
            new_config["sample_size"] = 64
            unet._internal_dict = FrozenDict(new_config)

        self.register_modules(
            vae=vae,
            audio_encoder=audio_encoder,
            unet=unet,
            scheduler=scheduler,
        )

        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)

        self.set_progress_bar_config(desc="Steps")

    def enable_vae_slicing(self):
        self.vae.enable_slicing()

    def disable_vae_slicing(self):
        self.vae.disable_slicing()

    @property
    def _execution_device(self):
        if self.device != torch.device("meta") or not hasattr(self.unet, "_hf_hook"):
            return self.device
        for module in self.unet.modules():
            if (
                hasattr(module, "_hf_hook")
                and hasattr(module._hf_hook, "execution_device")
                and module._hf_hook.execution_device is not None
            ):
                return torch.device(module._hf_hook.execution_device)
        return self.device

    def decode_latents(self, latents):
        latents = latents / self.vae.config.scaling_factor + self.vae.config.shift_factor
        latents = rearrange(latents, "b c f h w -> (b f) c h w")
        decoded_latents = self.vae.decode(latents).sample
        return decoded_latents

    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def check_inputs(self, height, width, callback_steps):
        assert height == width, "Height and width must be equal"

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

    def prepare_latents(self, num_frames, num_channels_latents, height, width, dtype, device, generator):
        shape = (
            1,
            num_channels_latents,
            1,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )  # (b, c, f, h, w)
        rand_device = "cpu" if device.type == "mps" else device
        latents = torch.randn(shape, generator=generator, device=rand_device, dtype=dtype).to(device)
        latents = latents.repeat(1, 1, num_frames, 1, 1)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def prepare_mask_latents(
        self, mask, masked_image, height, width, dtype, device, generator, do_classifier_free_guidance
    ):
        # resize the mask to latents shape as we concatenate the mask to the latents
        # we do that before converting to dtype to avoid breaking in case we're using cpu_offload
        # and half precision
        mask = torch.nn.functional.interpolate(
            mask, size=(height // self.vae_scale_factor, width // self.vae_scale_factor)
        )
        masked_image = masked_image.to(device=device, dtype=dtype)

        # encode the mask image into latents space so we can concatenate it to the latents
        masked_image_latents = self.vae.encode(masked_image).latent_dist.sample(generator=generator)
        masked_image_latents = (masked_image_latents - self.vae.config.shift_factor) * self.vae.config.scaling_factor

        # aligning device to prevent device errors when concating it with the latent model input
        masked_image_latents = masked_image_latents.to(device=device, dtype=dtype)
        mask = mask.to(device=device, dtype=dtype)

        # assume batch size = 1
        mask = rearrange(mask, "f c h w -> 1 c f h w")
        masked_image_latents = rearrange(masked_image_latents, "f c h w -> 1 c f h w")

        mask = torch.cat([mask] * 2) if do_classifier_free_guidance else mask
        masked_image_latents = (
            torch.cat([masked_image_latents] * 2) if do_classifier_free_guidance else masked_image_latents
        )
        return mask, masked_image_latents

    def prepare_image_latents(self, images, device, dtype, generator, do_classifier_free_guidance):
        images = images.to(device=device, dtype=dtype)
        image_latents = self.vae.encode(images).latent_dist.sample(generator=generator)
        image_latents = (image_latents - self.vae.config.shift_factor) * self.vae.config.scaling_factor
        image_latents = rearrange(image_latents, "f c h w -> 1 c f h w")
        image_latents = torch.cat([image_latents] * 2) if do_classifier_free_guidance else image_latents

        return image_latents

    def set_progress_bar_config(self, **kwargs):
        if not hasattr(self, "_progress_bar_config"):
            self._progress_bar_config = {}
        self._progress_bar_config.update(kwargs)

    @staticmethod
    def paste_surrounding_pixels_back(decoded_latents, pixel_values, masks, device, weight_dtype):
        # Paste the surrounding pixels back, because we only want to change the mouth region
        pixel_values = pixel_values.to(device=device, dtype=weight_dtype)
        masks = masks.to(device=device, dtype=weight_dtype)
        combined_pixel_values = decoded_latents * masks + pixel_values * (1 - masks)
        return combined_pixel_values

    @staticmethod
    def pixel_values_to_images(pixel_values: torch.Tensor):
        pixel_values = rearrange(pixel_values, "f c h w -> f h w c")
        pixel_values = (pixel_values / 2 + 0.5).clamp(0, 1)
        images = (pixel_values * 255).to(torch.uint8)
        images = images.cpu().numpy()
        return images

    def affine_transform_video(self, video_frames: np.ndarray):
        faces = []
        boxes = []
        affine_matrices = []
        print(f"Affine transforming {len(video_frames)} faces...")
        for frame in tqdm.tqdm(video_frames):
            face, box, affine_matrix = self.image_processor.affine_transform(frame)
            faces.append(face)
            boxes.append(box)
            affine_matrices.append(affine_matrix)

        faces = torch.stack(faces)
        return faces, boxes, affine_matrices

    def restore_video(
        self,
        faces: torch.Tensor,
        video_frames: np.ndarray,
        boxes: list,
        affine_matrices: list,
        batch_size: int = 1,
    ):
        video_frames = video_frames[: len(faces)]
        if batch_size > 1 and all(list(box) == list(boxes[0]) for box in boxes[: len(faces)]):
            # Batched path: the stock loop below does three host<->device round trips, a GPU sync and a
            # CPU erosion per frame, which leaves the GPU idle and makes this stage depend on how fast
            # (and how busy) the host CPU is. Here a whole batch stays on the GPU.
            print(f"Restoring {len(faces)} faces (batch size {batch_size})...")
            restorer = self.image_processor.restorer
            x1, y1, x2, y2 = boxes[0]
            size = (int(y2 - y1), int(x2 - x1))
            out_frames = np.empty((len(faces),) + video_frames.shape[1:], dtype=np.uint8)
            for start in tqdm.trange(0, len(faces), batch_size):
                end = min(start + batch_size, len(faces))
                batch_faces = torchvision.transforms.functional.resize(
                    faces[start:end], size=size, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True
                )
                matrices = torch.cat(
                    [m if torch.is_tensor(m) else torch.from_numpy(m).unsqueeze(0) for m in affine_matrices[start:end]]
                ).to(device=restorer.device, dtype=restorer.dtype)
                out_frames[start:end] = restorer.restore_imgs(
                    np.ascontiguousarray(video_frames[start:end]), batch_faces, matrices
                )
            return out_frames

        out_frames = []
        print(f"Restoring {len(faces)} faces...")
        for index, face in enumerate(tqdm.tqdm(faces)):
            x1, y1, x2, y2 = boxes[index]
            height = int(y2 - y1)
            width = int(x2 - x1)
            face = torchvision.transforms.functional.resize(
                face, size=(height, width), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True
            )
            out_frame = self.image_processor.restorer.restore_img(video_frames[index], face, affine_matrices[index])
            out_frames.append(out_frame)
        return np.stack(out_frames, axis=0)

    # ------------------------------------------------------------------
    # Reference cache
    #
    # Everything computed from the reference video alone (decoded frames, per-frame face
    # detection + affine alignment) is independent of the audio, yet the stock pipeline
    # redoes it on every call. When the same reference video is reused (an avatar, a
    # dubbing source, a serving process), that is most of the wall-clock time:
    # on an RTX 4090, 11.8s of audio took 116s of which only ~15s was diffusion.
    #
    # The affine results are memoised in memory and (optionally) on disk, keyed by the
    # video file identity (path + mtime + size), the resolution and the frame count.
    # Re-uploading/overwriting the video changes the key, so the cache self-invalidates.
    # ------------------------------------------------------------------

    _REF_FRAMES_CACHE_SIZE = 2  # decoded frames are large (~1GB for 15s of 720p)
    _REF_AFFINE_CACHE_SIZE = 4

    @staticmethod
    def _ref_cache_key(video_path: str, resolution, num_frames: int) -> str:
        st = os.stat(video_path)
        raw = f"v1|{os.path.abspath(video_path)}|{st.st_mtime_ns}|{st.st_size}|{resolution}|{num_frames}"
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:20]

    def _read_video_cached(self, video_path: str) -> np.ndarray:
        cache = self.__dict__.setdefault("_ref_frames_cache", {})
        st = os.stat(video_path)
        key = (os.path.abspath(video_path), st.st_mtime_ns, st.st_size)
        if key not in cache:
            while len(cache) >= self._REF_FRAMES_CACHE_SIZE:
                cache.pop(next(iter(cache)))
            cache[key] = read_video(video_path, use_decord=False)
        return cache[key]

    def _affine_transform_video_cached(self, video_frames: np.ndarray, key: str, cache_dir: Optional[str]):
        """affine_transform_video() over the WHOLE reference video, memoised in memory and on disk."""
        cache = self.__dict__.setdefault("_ref_affine_cache", {})
        if key in cache:
            print(f"Reference cache hit (memory): {len(video_frames)} frames")
            return cache[key]

        restorer = self.image_processor.restorer
        cache_file = os.path.join(cache_dir, f"{key}.pt") if cache_dir else None
        result = None
        if cache_file and os.path.exists(cache_file):
            try:
                data = torch.load(cache_file, map_location="cpu", weights_only=True)
                affine = data["affine_matrices"].to(device=restorer.device, dtype=restorer.dtype)
                result = (data["faces"], data["boxes"], [m.unsqueeze(0) for m in affine])
                print(f"Reference cache hit (disk): {len(video_frames)} frames <- {cache_file}")
            except Exception as e:  # corrupt / incompatible cache file: recompute
                print(f"Reference cache file unreadable ({e}); recomputing")

        if result is None:
            # The aligner smooths the affine bias over time (p_bias). Start from a clean state so the
            # result equals a fresh stock run, even when the ImageProcessor is reused across videos.
            restorer.p_bias = None
            result = self.affine_transform_video(video_frames)
            if cache_file:
                faces, boxes, affine_matrices = result
                os.makedirs(cache_dir, exist_ok=True)
                tmp_file = cache_file + ".tmp"
                torch.save(
                    {"faces": faces, "boxes": boxes, "affine_matrices": torch.cat(affine_matrices).cpu()},
                    tmp_file,
                )
                os.replace(tmp_file, cache_file)

        while len(cache) >= self._REF_AFFINE_CACHE_SIZE:
            cache.pop(next(iter(cache)))
        cache[key] = result
        return result

    def loop_video(self, whisper_chunks: list, video_frames: np.ndarray, cache_key=None, cache_dir=None):
        if cache_key is not None:
            # Cached path: align the whole reference once, then slice / ping-pong the cached results.
            # Alignment is causal (frame i depends only on frames <= i), so slicing the full-video
            # result is identical to the stock behaviour of aligning only the first N frames.
            faces, boxes, affine_matrices = self._affine_transform_video_cached(video_frames, cache_key, cache_dir)
            n = len(whisper_chunks)
            if n > len(video_frames):
                num_loops = math.ceil(n / len(video_frames))
                loop_video_frames, loop_faces, loop_boxes, loop_affine_matrices = [], [], [], []
                for i in range(num_loops):
                    if i % 2 == 0:
                        loop_video_frames.append(video_frames)
                        loop_faces.append(faces)
                        loop_boxes += boxes
                        loop_affine_matrices += affine_matrices
                    else:
                        loop_video_frames.append(video_frames[::-1])
                        loop_faces.append(faces.flip(0))
                        loop_boxes += boxes[::-1]
                        loop_affine_matrices += affine_matrices[::-1]
                return (
                    np.concatenate(loop_video_frames, axis=0)[:n],
                    torch.cat(loop_faces, dim=0)[:n],
                    loop_boxes[:n],
                    loop_affine_matrices[:n],
                )
            return video_frames[:n], faces[:n], boxes[:n], affine_matrices[:n]

        # If the audio is longer than the video, we need to loop the video
        if len(whisper_chunks) > len(video_frames):
            faces, boxes, affine_matrices = self.affine_transform_video(video_frames)
            num_loops = math.ceil(len(whisper_chunks) / len(video_frames))
            loop_video_frames = []
            loop_faces = []
            loop_boxes = []
            loop_affine_matrices = []
            for i in range(num_loops):
                if i % 2 == 0:
                    loop_video_frames.append(video_frames)
                    loop_faces.append(faces)
                    loop_boxes += boxes
                    loop_affine_matrices += affine_matrices
                else:
                    loop_video_frames.append(video_frames[::-1])
                    loop_faces.append(faces.flip(0))
                    loop_boxes += boxes[::-1]
                    loop_affine_matrices += affine_matrices[::-1]

            video_frames = np.concatenate(loop_video_frames, axis=0)[: len(whisper_chunks)]
            faces = torch.cat(loop_faces, dim=0)[: len(whisper_chunks)]
            boxes = loop_boxes[: len(whisper_chunks)]
            affine_matrices = loop_affine_matrices[: len(whisper_chunks)]
        else:
            video_frames = video_frames[: len(whisper_chunks)]
            faces, boxes, affine_matrices = self.affine_transform_video(video_frames)

        return video_frames, faces, boxes, affine_matrices

    @torch.no_grad()
    def __call__(
        self,
        video_path: str,
        audio_path: str,
        video_out_path: str,
        num_frames: int = 16,
        video_fps: int = 25,
        audio_sample_rate: int = 16000,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 20,
        guidance_scale: float = 1.5,
        weight_dtype: Optional[torch.dtype] = torch.float16,
        eta: float = 0.0,
        mask_image_path: str = "latentsync/utils/mask.png",
        temp_dir: str = "temp",
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        ref_cache: bool = True,
        ref_cache_dir: Optional[str] = ".cache/ref_affine",
        legacy_encode: bool = False,
        restore_batch_size: int = 16,
        **kwargs,
    ):
        """
        legacy_encode: True restores the stock two-pass encoding (crf 13, then re-encode at crf 18).
        restore_batch_size: frames pasted back per GPU batch (~60MB of VRAM per 720p frame).
            1 restores the stock per-frame loop.
        ref_cache: reuse everything derived from the reference video alone (face detector, decoded
            frames, per-frame affine alignment) across calls. Set False for the stock behaviour.
        ref_cache_dir: where alignment results are persisted so that new processes (CLI runs, server
            restarts) also start warm. None keeps the cache in memory only.
        """
        is_train = self.unet.training
        self.unet.eval()

        check_ffmpeg_installed()

        # Per-stage wall-clock, printed at the end and kept in self.last_timings
        timings = {}
        stage_start = [time.perf_counter()]

        def mark(stage):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            now = time.perf_counter()
            timings[stage] = timings.get(stage, 0.0) + now - stage_start[0]
            stage_start[0] = now

        # 0. Define call parameters
        device = self._execution_device
        mask_image = load_fixed_mask(height, mask_image_path)
        reusable = getattr(self, "image_processor", None)
        if ref_cache and reusable is not None and reusable.resolution == height:
            reusable.mask_image = mask_image  # keep the loaded face detector
        else:
            self.image_processor = ImageProcessor(height, device="cuda", mask_image=mask_image)
        self.set_progress_bar_config(desc=f"Sample frames: {num_frames}")

        # 1. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # 2. Check inputs
        self.check_inputs(height, width, callback_steps)

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. set timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 4. Prepare extra step kwargs.
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        mark("setup")
        whisper_feature = self.audio_encoder.audio2feat(audio_path)
        whisper_chunks = self.audio_encoder.feature2chunks(feature_array=whisper_feature, fps=video_fps)

        audio_samples = read_audio(audio_path)
        mark("audio")
        if ref_cache:
            video_frames = self._read_video_cached(video_path)
            cache_key = self._ref_cache_key(video_path, height, len(video_frames))
            video_frames, faces, boxes, affine_matrices = self.loop_video(
                whisper_chunks, video_frames, cache_key=cache_key, cache_dir=ref_cache_dir
            )
        else:
            video_frames = read_video(video_path, use_decord=False)
            video_frames, faces, boxes, affine_matrices = self.loop_video(whisper_chunks, video_frames)
        mark("reference")

        synced_video_frames = []

        num_channels_latents = self.vae.config.latent_channels

        # Prepare latent variables
        all_latents = self.prepare_latents(
            len(whisper_chunks),
            num_channels_latents,
            height,
            width,
            weight_dtype,
            device,
            generator,
        )

        num_inferences = math.ceil(len(whisper_chunks) / num_frames)
        for i in tqdm.tqdm(range(num_inferences), desc="Doing inference..."):
            if self.unet.add_audio_layer:
                audio_embeds = torch.stack(whisper_chunks[i * num_frames : (i + 1) * num_frames])
                audio_embeds = audio_embeds.to(device, dtype=weight_dtype)
                if do_classifier_free_guidance:
                    null_audio_embeds = torch.zeros_like(audio_embeds)
                    audio_embeds = torch.cat([null_audio_embeds, audio_embeds])
            else:
                audio_embeds = None
            inference_faces = faces[i * num_frames : (i + 1) * num_frames]
            latents = all_latents[:, :, i * num_frames : (i + 1) * num_frames]
            ref_pixel_values, masked_pixel_values, masks = self.image_processor.prepare_masks_and_masked_images(
                inference_faces, affine_transform=False
            )

            # 7. Prepare mask latent variables
            mask_latents, masked_image_latents = self.prepare_mask_latents(
                masks,
                masked_pixel_values,
                height,
                width,
                weight_dtype,
                device,
                generator,
                do_classifier_free_guidance,
            )

            # 8. Prepare image latents
            ref_latents = self.prepare_image_latents(
                ref_pixel_values,
                device,
                weight_dtype,
                generator,
                do_classifier_free_guidance,
            )

            # 9. Denoising loop
            num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
            with self.progress_bar(total=num_inference_steps) as progress_bar:
                for j, t in enumerate(timesteps):
                    # expand the latents if we are doing classifier free guidance
                    unet_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents

                    unet_input = self.scheduler.scale_model_input(unet_input, t)

                    # concat latents, mask, masked_image_latents in the channel dimension
                    unet_input = torch.cat([unet_input, mask_latents, masked_image_latents, ref_latents], dim=1)

                    # predict the noise residual
                    noise_pred = self.unet(unet_input, t, encoder_hidden_states=audio_embeds).sample

                    # perform guidance
                    if do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_audio = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_audio - noise_pred_uncond)

                    # compute the previous noisy sample x_t -> x_t-1
                    latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                    # call the callback, if provided
                    if j == len(timesteps) - 1 or ((j + 1) > num_warmup_steps and (j + 1) % self.scheduler.order == 0):
                        progress_bar.update()
                        if callback is not None and j % callback_steps == 0:
                            callback(j, t, latents)

            # Recover the pixel values
            decoded_latents = self.decode_latents(latents)
            decoded_latents = self.paste_surrounding_pixels_back(
                decoded_latents, ref_pixel_values, 1 - masks, device, weight_dtype
            )
            synced_video_frames.append(decoded_latents)

        mark("diffusion")
        synced_video_frames = self.restore_video(
            torch.cat(synced_video_frames), video_frames, boxes, affine_matrices, batch_size=restore_batch_size
        )
        mark("restore")

        audio_samples_remain_length = int(synced_video_frames.shape[0] / video_fps * audio_sample_rate)
        audio_samples = audio_samples[:audio_samples_remain_length].cpu().numpy()

        if is_train:
            self.unet.train()

        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        os.makedirs(temp_dir, exist_ok=True)

        sf.write(os.path.join(temp_dir, "audio.wav"), audio_samples, audio_sample_rate)

        if legacy_encode:
            # Stock: encode at crf 13, then re-encode the whole video at crf 18 while muxing the audio.
            write_video(os.path.join(temp_dir, "video.mp4"), synced_video_frames, fps=video_fps)
            command = f"ffmpeg -y -loglevel error -nostdin -i {os.path.join(temp_dir, 'video.mp4')} -i {os.path.join(temp_dir, 'audio.wav')} -c:v libx264 -crf 18 -c:a aac -q:v 0 -q:a 0 {video_out_path}"
        else:
            # Encode once at the final quality (crf 18) and only mux the audio: same target quality and
            # file size as stock, one x264 pass instead of two, and no second-generation encoding loss.
            write_video(os.path.join(temp_dir, "video.mp4"), synced_video_frames, fps=video_fps, crf=18)
            command = f"ffmpeg -y -loglevel error -nostdin -i {os.path.join(temp_dir, 'video.mp4')} -i {os.path.join(temp_dir, 'audio.wav')} -c:v copy -c:a aac -q:a 0 {video_out_path}"
        subprocess.run(command, shell=True)
        mark("encode")

        self.last_timings = dict(timings, total=sum(timings.values()))
        print("Timings: " + " | ".join(f"{stage} {seconds:.1f}s" for stage, seconds in self.last_timings.items()))
