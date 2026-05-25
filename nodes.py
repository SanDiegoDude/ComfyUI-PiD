"""
ComfyUI custom nodes for PiD (Pixel Diffusion Decoder).

PiD is a plug-and-play diffusion decoder that replaces VAE decoders,
turning latent representations directly into super-resolved pixels.

Nodes:
  - PiDDecode: Takes latent samples + prompt, decodes via PiD to produce
    high-resolution super-resolved images (e.g. 512px latent → 2048px output).
"""

import logging
import os
import sys

import torch

logger = logging.getLogger("PiD")

# ---------------------------------------------------------------------------
# Ensure the PiD repo root is on sys.path so `import pid` resolves.
# ComfyUI may or may not do this depending on version; be defensive.
# ---------------------------------------------------------------------------
_PID_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PID_ROOT not in sys.path:
    sys.path.insert(0, _PID_ROOT)

# ---------------------------------------------------------------------------
# Model cache — keyed by (backbone, ckpt_type) to avoid reloading.
# ---------------------------------------------------------------------------
_MODEL_CACHE: dict[tuple[str, str], tuple] = {}

# ---------------------------------------------------------------------------
# Prompt cache — keyed by (prompt, backbone, ckpt_type) to save text encoding time
# ---------------------------------------------------------------------------
_PROMPT_CACHE: dict[tuple[str, str, str], torch.Tensor] = {}

# ---------------------------------------------------------------------------
# Auto-download configuration.
#
# All PiD assets (the per-(backbone, ckpt_type) decoder .pth + the auxiliary
# VAE / RAE / Scale-RAE files used by the corresponding tokenizer) are hosted
# at https://huggingface.co/nvidia/PiD under a single `checkpoints/...` tree.
#
# We mirror that tree into one of two places, in this order of preference:
#   1. <ComfyUI>/models/PiD/checkpoints/...     (preferred — standard layout)
#   2. <repo>/checkpoints/...                   (legacy — for users who already
#                                               followed the upstream README's
#                                               huggingface-cli download command)
#
# The chosen root becomes the cwd during model load so the relative
# `./checkpoints/...` paths embedded in the upstream tokenizer code resolve
# correctly without us having to monkey-patch them.
# ---------------------------------------------------------------------------
_HF_REPO_ID = "nvidia/PiD"

# Auxiliary files each backbone needs in addition to its PiD decoder .pth.
# Paths are relative to the asset root (i.e. they include the leading
# `checkpoints/` segment to match the upstream HF repo layout).
_AUX_FILES_PER_BACKBONE: dict[str, tuple[str, ...]] = {
    "flux":      ("checkpoints/ae.safetensors",),
    "flux2":     ("checkpoints/flux2_ae.safetensors",),
    "sd3":       ("checkpoints/sd3_vae/vae/diffusion_pytorch_model.safetensors",),
    "zimage":    ("checkpoints/ae.safetensors",),  # reuses Flux1 VAE
    "rae": (
        "checkpoints/rae/decoders/dinov2/wReg_base/ViTXL_n08_i512/model.pt",
        "checkpoints/rae/stats/dinov2/wReg_base/imagenet1k_512/stat.pt",
    ),
    "scale_rae": (
        "checkpoints/scale_rae/decoder/siglip2_sop14_i224_web73M_ganw3_decXL.pt",
        "checkpoints/scale_rae/decoder/XL_decoder_config.json",
    ),
}


def _comfy_pid_models_dir():
    """Return <ComfyUI>/models/PiD, or None if ComfyUI's folder_paths isn't importable."""
    try:
        import folder_paths
    except ImportError:
        return None
    return os.path.join(folder_paths.models_dir, "PiD")


_MODEL_FILE_SUFFIXES = (".pth", ".pt", ".safetensors")


def _has_model_files(root: str) -> bool:
    """True iff `root` contains at least one .pth/.pt/.safetensors anywhere under it.

    Used to detect a populated legacy `<repo>/checkpoints/` tree without being
    fooled by the empty placeholder directories the upstream repo ships.
    """
    if not os.path.isdir(root):
        return False
    for _dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            if name.endswith(_MODEL_FILE_SUFFIXES):
                return True
    return False


def _resolve_asset_root() -> str:
    """Choose where PiD assets live (and will be downloaded into).

    Prefers <ComfyUI>/models/PiD. Falls back to the legacy <repo>/checkpoints/
    layout only when the user already populated it (so we don't force a
    re-download for people who followed the upstream README).

    The upstream repo ships empty placeholder directories under
    `<repo>/checkpoints/` — only treat that path as "populated" once it
    actually contains a model file, not just empty subdirs.
    """
    legacy_ckpt_dir = os.path.join(_PID_ROOT, "checkpoints")
    if _has_model_files(legacy_ckpt_dir):
        return _PID_ROOT

    comfy_root = _comfy_pid_models_dir()
    if comfy_root is not None:
        os.makedirs(comfy_root, exist_ok=True)
        return comfy_root

    # No ComfyUI folder_paths available (e.g. running outside Comfy) — use
    # the repo root and let hf_hub_download create the checkpoints/ tree there.
    return _PID_ROOT


def _ensure_asset(asset_root: str, rel_path: str) -> str:
    """Return the absolute path to `rel_path` under `asset_root`, downloading it
    from `nvidia/PiD` on Hugging Face if it isn't already on disk.
    """
    target = os.path.join(asset_root, rel_path)
    if os.path.exists(target):
        return target

    from huggingface_hub import hf_hub_download

    os.makedirs(os.path.dirname(target), exist_ok=True)
    logger.info(
        f"PiD: '{rel_path}' not found under {asset_root}; "
        f"downloading from huggingface.co/{_HF_REPO_ID} (one-time, may take a while)…"
    )
    return hf_hub_download(
        repo_id=_HF_REPO_ID,
        filename=rel_path,
        local_dir=asset_root,
    )


def _ensure_pid_assets(backbone: str, ckpt_type: str) -> tuple[str, str]:
    """Make sure every file PiD needs for (backbone, ckpt_type) is on disk.

    Returns (asset_root, absolute_path_to_pid_decoder_pth).
    """
    from pid._src.inference.checkpoint_registry import get_pid_checkpoint

    ckpt_info = get_pid_checkpoint(backbone, ckpt_type)
    asset_root = _resolve_asset_root()

    decoder_path = _ensure_asset(asset_root, ckpt_info.checkpoint_path)
    for aux in _AUX_FILES_PER_BACKBONE.get(backbone, ()):
        _ensure_asset(asset_root, aux)

    return asset_root, decoder_path


def _get_available_backbones():
    """Return the list of backbone + ckpt_type combos that have checkpoints present."""
    from pid._src.inference.checkpoint_registry import PID_CHECKPOINT_REGISTRY

    asset_root = _resolve_asset_root()
    available = []
    for (backbone, ckpt_type), ckpt_info in PID_CHECKPOINT_REGISTRY.items():
        ckpt_path = os.path.join(asset_root, ckpt_info.checkpoint_path)
        if os.path.exists(ckpt_path):
            available.append(f"{backbone}_{ckpt_type}")
    return available


def _load_pid_model(backbone: str, ckpt_type: str):
    """Load a PiD model, returning (model, config). Uses cache."""
    cache_key = (backbone, ckpt_type)
    if cache_key in _MODEL_CACHE:
        logger.info(f"PiD model cache hit: {backbone}/{ckpt_type}")
        return _MODEL_CACHE[cache_key]

    logger.info(f"Loading PiD model: backbone={backbone}, ckpt_type={ckpt_type} ...")

    from pid._src.inference.checkpoint_registry import get_pid_checkpoint
    from pid._src.utils.model_loader import load_model_from_checkpoint

    ckpt_info = get_pid_checkpoint(backbone, ckpt_type)
    experiment_name = ckpt_info.experiment

    # Resolve assets — downloads any missing decoder/auxiliary files from
    # huggingface.co/nvidia/PiD into <ComfyUI>/models/PiD (or, if the user
    # pre-populated the legacy <repo>/checkpoints/ layout, into that).
    asset_root, checkpoint_path = _ensure_pid_assets(backbone, ckpt_type)

    config_file = os.path.join(_PID_ROOT, "pid", "_src", "configs", "pid", "config.py")

    # The upstream tokenizer code (flux_vae.py, dinov2_vae.py, etc.) hardcodes
    # auxiliary VAE/RAE paths as `./checkpoints/...`, resolved against cwd. To
    # honor that without monkey-patching the tokenizers, run the load with cwd
    # set to the asset root. Restore in finally so we don't leak chdir state
    # back to ComfyUI's worker.
    _prev_cwd = os.getcwd()
    try:
        os.chdir(asset_root)
        model, config = load_model_from_checkpoint(
            experiment_name=experiment_name,
            checkpoint_path=checkpoint_path,
            config_file=config_file,
            enable_fsdp=False,
            experiment_opts=[],
            strict=False,
            load_ema_to_reg=False,
        )
    finally:
        os.chdir(_prev_cwd)
    model.eval()

    # NOTE: an earlier version tried to apply channels_last to model.vae_encoder
    # here. It logged "Failed to optimize VAE memory layout" because the
    # vae_encoder attribute holds a FluxVAEInterface / DinoV2VAEInterface /
    # etc. wrapper (not an nn.Module — the actual nn.Module sits at
    # model.vae_encoder.model.model). More importantly, the released distill
    # checkpoints all have lq_condition_type="latent" (see
    # pid/_src/configs/pid/experiment/shared_config.py), so the vae_encoder is
    # never invoked at inference: PiD reads the pre-encoded LQ_latent straight
    # from the data batch and the image branch of the LQ projection is unused.
    # Optimizing a forward pass that never runs is wasted work, so the block
    # has been removed. spatial_compression_factor (read in decode() to size
    # the output) is a property, not a forward, and works without any of this.

    # Offload to CPU initially to save VRAM when not executing
    model.to("cpu")
    if hasattr(model, "text_encoder") and model.text_encoder is not None:
        model.text_encoder.to("cpu")
    if hasattr(model, "_null_caption_embs") and isinstance(model._null_caption_embs, torch.Tensor):
        model._null_caption_embs = model._null_caption_embs.to("cpu")

    _MODEL_CACHE[cache_key] = (model, config)
    logger.info(f"PiD model loaded successfully (cached on CPU): {backbone}/{ckpt_type}")
    return model, config


# ---------------------------------------------------------------------------
# Helper classes for ComfyUI sampler/scheduler integration
# ---------------------------------------------------------------------------
class MockModelSampling:
    def __init__(self):
        self.sigma_min = 0.0
        self.sigma_max = 1.0
        self.sigmas = torch.linspace(1.0, 0.0, 1000)

    def timestep(self, sigma):
        return sigma * 999.0

    def sigma(self, timestep):
        return timestep / 999.0

    def noise_scaling(self, sigma, noise, latent_image, max_denoise=False):
        return noise

    def inverse_noise_scaling(self, sigma, samples):
        return samples


class MockInnerInnerModel:
    def scale_latent_inpaint(self, *args, **kwargs):
        return 0.0


class MockInnerModel:
    def __init__(self):
        self.model_sampling = MockModelSampling()
        self.inner_model = MockInnerInnerModel()


class MockModelWrap:
    def __init__(self, denoise_fn):
        self.inner_model = MockInnerModel()
        self.denoise_fn = denoise_fn

    def __call__(self, x, sigma, model_options={}, seed=None):
        return self.denoise_fn(x, sigma)

    def get_model_object(self, name):
        if name == "model_sampling":
            return self.inner_model.model_sampling
        return None


class PiDDecode:
    """
    Decode latent samples using PiD (Pixel Diffusion Decoder).

    PiD replaces the standard VAE decoder with a conditional pixel-space
    diffusion model that produces super-resolved images in one pass.
    For example, a 512px Flux latent becomes a 2048px image (4× SR).

    Supported backbones:
      - flux: Flux 1 (16-ch VAE, 8× spatial compression)
      - flux2: Flux 2 (128-ch VAE, 16× spatial compression)
      - sd3: Stable Diffusion 3 (16-ch VAE, 8× spatial compression)
      - zimage: ZImage (reuses Flux 1's VAE)
      - rae: Representation Autoencoder / DINOv2 (768-ch RAE, 16× spatial compression)
      - scale_rae: Scale RAE / SigLIP-2 (768-ch RAE, 16× spatial compression)

    Supported checkpoint variants:
      - 2k: Original 2048px-trained decoder (512→2048 at 4× scale, or 256→2048 at 8× scale)
      - 2kto4k: Multi-resolution decoder (1024→4K at 4× scale)
    """

    @classmethod
    def INPUT_TYPES(cls):
        try:
            import comfy.samplers
            samplers = ["default"] + sorted(list(comfy.samplers.KSampler.SAMPLERS))
            schedulers = ["default"] + sorted(list(comfy.samplers.KSampler.SCHEDULERS))
        except Exception:
            samplers = ["default", "ode_euler", "ode_heun", "sde_ancestral"]
            schedulers = ["default", "linear", "karras", "exponential", "cosine"]

        return {
            "required": {
                "latent": ("LATENT",),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": (
                        "Text prompt describing the IMAGE CONTENT (not style modifiers). "
                        "PiD wraps this in a chi-prompt template that asks gemma-2-2b to expand it into a "
                        "detailed visual description, so generic placeholders like 'high quality, detailed' "
                        "give gemma nothing to anchor on and produce mushy / over-smoothed output. "
                        "For best quality, pass the same prompt you used to generate the latent. "
                        "Empty string is allowed but quality will suffer."
                    ),
                }),
                "backbone": (["flux", "flux2", "sd3", "zimage", "rae", "scale_rae"], {
                    "default": "flux",
                    "tooltip": "VAE backbone. Must match the VAE/encoder used to encode the latent. zimage reuses flux's VAE.",
                }),
                "ckpt_type": (["2k", "2kto4k"], {
                    "default": "2k",
                    "tooltip": "2k: 512→2048px decoder. 2kto4k: 1024→4K decoder.",
                }),
                "match_original_size": ("BOOLEAN", {
                    "default": True,
                    "tooltip": (
                        "When True (default), PiD acts as a drop-in VAE alternative — output "
                        "matches the size a regular VAE Decode would have produced. The latent "
                        "is downsampled toward the model's training grid (clamped: never below "
                        "natural grid, never above your input), PiD runs at its training output "
                        "resolution, and the final image is bilinear-resized to VAE-native size. "
                        "Keeps PiD inside its training distribution regardless of input resolution. "
                        "When False, PiD upscales — output = VAE_native × scale (true SR mode)."
                    ),
                }),
                "scale": ("INT", {
                    "default": 4,
                    "min": 1,
                    "max": 8,
                    "step": 1,
                    "tooltip": (
                        "PiD's SR factor. The released checkpoints are sr4x (Flux/SD3/Flux2/RAE) "
                        "or sr8x (scale_rae) — change this only if you know what you're doing. "
                        "match_original_size=True: how aggressively to downsample the latent "
                        "before reconstruction (capped at the model's training grid). "
                        "match_original_size=False: the literal output upscale factor "
                        "(output = VAE_native × scale)."
                    ),
                }),
                "pid_inference_steps": ("INT", {
                    "default": 4,
                    "min": 1,
                    "max": 20,
                    "step": 1,
                    "tooltip": "Number of denoising steps for PiD. 4 is optimal for the distilled checkpoints.",
                }),
                "cfg_scale": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 15.0,
                    "step": 0.1,
                    "tooltip": "Classifier-free guidance scale for PiD.",
                }),
                "degrade_sigma": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Noise level indicator. 0.0 = clean latent (standard KSampler output). "
                               "Higher values indicate the latent has more noise (e.g. from early termination).",
                }),
                "vram_mode": (["gpu", "cpu"], {
                    "default": "gpu",
                    "tooltip": "gpu: Run on the active CUDA device. cpu: Run on CPU (very slow; only use if you don't have a usable GPU).",
                }),
                "shift": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 20.0,
                    "step": 0.1,
                    "tooltip": "Noise schedule shift. 0.0 uses config/model defaults. Larger shifts delay noise removal (smoother/detailed); smaller shifts accelerate it (sharper).",
                }),
                "sampler": (samplers, {
                    "default": "default",
                    "tooltip": "Sampler algorithm to use for the diffusion decoding loop. 'default' uses the model config's standard sampler.",
                }),
                "scheduler": (schedulers, {
                    "default": "default",
                    "tooltip": "Noise schedule type. 'default' uses the model's pre-configured steps. Others generate dynamic step distributions.",
                }),
                "precision": (["model_default", "fp16", "bf16", "fp32"], {
                    "default": "model_default",
                    "tooltip": "Computation precision. 'model_default' uses the checkpoint's native precision (usually bf16). fp16 is faster on older GPUs.",
                }),
                "compile_mode": (["none", "reduce-overhead", "max-autotune"], {
                    "default": "none",
                    "tooltip": "Compile the PixDiT model using torch.compile. 'reduce-overhead' speeds up inference but has a 1-2 minute warmup on the first run.",
                }),
                "prompt_cache": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Cache prompt embeddings in system RAM. Subsequent runs with the same prompt will bypass the text encoder entirely, saving time and VRAM.",
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xFFFFFFFF,
                    "tooltip": "Random seed for the PiD diffusion process.",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "decode"
    CATEGORY = "PiD"
    DESCRIPTION = (
        "Decode latent samples using PiD (Pixel Diffusion Decoder). "
        "Replaces standard VAE Decode with a diffusion-based decoder that "
        "produces super-resolved images (e.g. 512px → 2048px)."
    )

    def decode(
        self,
        latent: dict,
        prompt: str,
        backbone: str,
        ckpt_type: str,
        scale: int,
        pid_inference_steps: int,
        cfg_scale: float,
        seed: int,
        match_original_size: bool = True,
        degrade_sigma: float = 0.0,
        vram_mode: str = "gpu",
        shift: float = 0.0,
        sampler: str = "default",
        scheduler: str = "default",
        precision: str = "model_default",
        compile_mode: str = "none",
        prompt_cache: bool = True,
    ):
        # Backwards compat: an earlier version of this node exposed a 3-way
        # "low / high / cpu" vram_mode. The "low" variant tried to keep VRAM
        # down by offloading the text encoder during sampling, but that path
        # actually moved a multi-GB text encoder to CPU only to immediately
        # move it back to GPU via model.to(device), tripling the transfer
        # cost for no actual VRAM savings. Map the old values onto "gpu" so
        # workflows saved with the previous menu still work.
        if vram_mode in ("low", "high"):
            vram_mode = "gpu"

        # ---- Load model (cached) ----
        model, config = _load_pid_model(backbone, ckpt_type)

        # ---- Apply Precision Casting ----
        orig_autocast = model.autocast_dtype
        if precision == "fp16":
            model.to(dtype=torch.float16)
            model.autocast_dtype = torch.float16
        elif precision == "bf16":
            model.to(dtype=torch.bfloat16)
            model.autocast_dtype = torch.bfloat16
        elif precision == "fp32":
            model.to(dtype=torch.float32)
            model.autocast_dtype = None

        # ---- Compile PixDiT Network ----
        if compile_mode != "none" and not hasattr(model.net, "_orig_mod"):
            logger.info(f"PiD: Compiling PixDiT network with mode='{compile_mode}'...")
            try:
                model.net = torch.compile(model.net, mode=compile_mode)
                logger.info("PiD: Network compiled successfully!")
            except Exception as e:
                logger.warning(f"Failed to compile network: {e}")

        import comfy.model_management
        import comfy.utils

        # Determine target device
        if vram_mode == "cpu":
            device = torch.device("cpu")
        else:
            device = comfy.model_management.get_torch_device()

        # Unload ComfyUI's internal models from GPU if running on GPU
        if device.type == "cuda":
            logger.info("PiD: Unloading ComfyUI models from GPU to free memory...")
            comfy.model_management.unload_all_models()

        # ---- Extract latent tensor from ComfyUI format ----
        # ComfyUI latents: {"samples": tensor} where tensor is [B, C, H, W]
        latent_tensor = latent["samples"]
        B, C, zH, zW = latent_tensor.shape

        logger.info(
            f"PiD decode: latent shape={latent_tensor.shape}, backbone={backbone}, "
            f"ckpt_type={ckpt_type}, scale={scale}, steps={pid_inference_steps}, "
            f"degrade_sigma={degrade_sigma}, vram_mode={vram_mode}, shift={shift}, "
            f"precision={precision}, compile={compile_mode}, cache={prompt_cache}"
        )

        # ---- Determine VAE spatial compression factor dynamically ----
        if hasattr(model, "vae_encoder") and model.vae_encoder is not None:
            vae_compression = getattr(model.vae_encoder, "spatial_compression_factor", 8)
            if callable(vae_compression):
                try:
                    vae_compression = vae_compression()
                except Exception:
                    pass
        else:
            # Fallback based on backbone name if vae_encoder isn't present
            if backbone in ("flux", "sd3", "zimage"):
                vae_compression = 8
            elif backbone in ("flux2", "rae", "scale_rae"):
                vae_compression = 16
            else:
                vae_compression = 8

        # Cache the original latent dimensions before we (maybe) downscale them
        # so log lines and the LQ_video_or_image / target_h placeholders that
        # follow can refer to the version PiD will actually consume.
        orig_zH, orig_zW = zH, zW

        # ---- Compute the model's "natural" training latent grid ----
        # Each released checkpoint was distilled at a specific (output_res,
        # vae_compression, sr_scale_trained) triple. PiD's network has those
        # ratios baked into the LQ-projection layer (z_to_patch_ratio) so the
        # model's training distribution sits at exactly:
        #     latent_grid = output_res / (vae_compression * sr_scale_trained)
        # Going noticeably below that grid (e.g. by floor-dividing a small
        # latent through a large `scale`) feeds the LQ projection a token
        # count it never saw in training, and the output goes mushy/soft —
        # which is the most common cause of "PiD output looks bad" reports.
        sr_scale_trained = 8 if backbone == "scale_rae" else 4
        training_output_res = 4096 if ckpt_type == "2kto4k" else 2048
        natural_grid_h = max(1, training_output_res // (vae_compression * sr_scale_trained))
        natural_grid_w = natural_grid_h  # released checkpoints all train square

        # Final user-facing output size. For match_original_size=True this is
        # the VAE-native size of the original latent; for SR mode it's
        # latent_grid * vae_compression * scale.
        if match_original_size:
            # ---- VAE-alternative mode (default) ----
            # Goal: emit an image at exactly the size a plain VAE.decode would
            # have produced. We do this in three steps:
            #   1. Pick a PiD input grid that's <= the user's latent (no
            #      magic upsampling) and >= the model's natural training grid
            #      (so the LQ projection sees a token count it has actually
            #      seen during training). The user's `scale` knob is the
            #      target divisor; `natural_grid_h` is the floor.
            #   2. Run PiD at its training-time output resolution (pid_grid *
            #      vae_compression * sr_scale_trained). This keeps every
            #      RoPE / lq-projection / fm-shift assumption inside the
            #      model's distilled regime.
            #   3. Bilinear-resize PiD's output to the user-requested
            #      VAE-native size. This last step is a small, lossy-only-
            #      because-it's-a-resize step — much cheaper than feeding
            #      the network out-of-distribution token counts.
            user_target_h = orig_zH * vae_compression
            user_target_w = orig_zW * vae_compression

            pid_zH = min(orig_zH, max(orig_zH // max(1, scale), natural_grid_h))
            pid_zW = min(orig_zW, max(orig_zW // max(1, scale), natural_grid_w))

            if (pid_zH, pid_zW) != (orig_zH, orig_zW):
                import torch.nn.functional as F
                latent_tensor = F.interpolate(
                    latent_tensor, size=(pid_zH, pid_zW), mode="area",
                )
            zH, zW = pid_zH, pid_zW

            pid_target_h = zH * vae_compression * sr_scale_trained
            pid_target_w = zW * vae_compression * sr_scale_trained
            image_size = (pid_target_h, pid_target_w)

            final_target_h = user_target_h
            final_target_w = user_target_w
            need_post_resize = (pid_target_h, pid_target_w) != (final_target_h, final_target_w)

            logger.info(
                f"PiD decode (match_original_size=True): "
                f"latent {orig_zH}×{orig_zW} → PiD-input {zH}×{zW} "
                f"(natural training grid={natural_grid_h}, sr_scale_trained={sr_scale_trained}) "
                f"→ PiD output {pid_target_h}×{pid_target_w} "
                f"→ final {final_target_h}×{final_target_w} (VAE-native of original latent)"
            )
        else:
            # ---- Super-resolution mode ----
            # Keep the latent at its full grid and ask PiD for a `scale`-times
            # larger output. NTK-aware RoPE handles output sizes above the
            # training resolution gracefully; below sr_scale_trained the LQ
            # projection still architecturally fits but expect quality drop.
            pid_target_h = zH * vae_compression * scale
            pid_target_w = zW * vae_compression * scale
            image_size = (pid_target_h, pid_target_w)
            final_target_h = pid_target_h
            final_target_w = pid_target_w
            need_post_resize = False
            logger.info(
                f"PiD decode (match_original_size=False, SR×{scale}): "
                f"latent {zH}×{zW} → vae_native {zH * vae_compression}×{zW * vae_compression} "
                f"→ PiD output {pid_target_h}×{pid_target_w}"
            )

        # ---- Move model to target device (one-shot) ----
        # Past versions tried to keep VRAM down by bouncing the text encoder
        # in and out of GPU. That ended up moving the same multi-GB tensor
        # three times per call without saving anything (model.to(device) at
        # the end of the encoding phase pulled the text encoder back onto GPU
        # anyway because it's a submodule). Just put everything on the target
        # device once; the post-decode cleanup below offloads it back to CPU
        # to free VRAM for downstream nodes.
        model.to(device)
        if hasattr(model, "text_encoder") and model.text_encoder is not None:
            model.text_encoder.to(device)
        if hasattr(model, "_null_caption_embs") and isinstance(model._null_caption_embs, torch.Tensor):
            model._null_caption_embs = model._null_caption_embs.to(device)

        # ---- Text Encoding Phase ----
        caption_key = model.config.input_caption_key
        captions = [prompt] * B

        # Check Prompt Cache
        prompt_cache_key = (prompt, backbone, ckpt_type)
        if prompt_cache and prompt_cache_key in _PROMPT_CACHE:
            logger.info("PiD: Prompt embedding cache hit! Bypassing text encoder.")
            caption_embs = _PROMPT_CACHE[prompt_cache_key].to(device)
        else:
            with torch.no_grad():
                caption_embs, _ = model._encode_text_raw(captions)
            if prompt_cache:
                _PROMPT_CACHE[prompt_cache_key] = caption_embs.cpu()

        # ---- Diffusion Sampling Phase ----
        data_batch = {
            caption_key: captions,
            "caption_embs": caption_embs,
            "LQ_latent": latent_tensor.to(dtype=model.autocast_dtype if model.autocast_dtype else torch.float32, device=device),
            # The released distill checkpoints have lq_condition_type="latent"
            # (see pid/_src/configs/pid/experiment/shared_config.py), so the
            # network's image branch is unused. Pass a small zero placeholder
            # to satisfy the data-batch contract — _demo_from_clean_common.py
            # in the upstream repo uses the same zeros placeholder for the
            # exact same reason. We size it at the latent's vae-native size
            # (cheap allocation) instead of the SR target so we don't waste
            # memory on a tensor the model immediately discards.
            "LQ_video_or_image": torch.zeros(
                B, 3, zH * vae_compression, zW * vae_compression,
                dtype=model.autocast_dtype if model.autocast_dtype else torch.float32, device=device,
            ).to(memory_format=torch.channels_last),
            "degrade_sigma": torch.tensor(
                [float(degrade_sigma)] * B, device=device, dtype=torch.float32,
            ),
        }

        # Resolve shift (0.0 means config / model defaults)
        shift_val = float(shift) if float(shift) > 0.0 else None

        native_samplers = ["default", "ode_euler", "ode_heun", "sde_ancestral"]
        native_schedulers = ["default", "linear", "karras", "exponential", "cosine"]

        is_native = (sampler in native_samplers) and (scheduler in native_schedulers)

        if is_native:
            pbar = comfy.utils.ProgressBar(pid_inference_steps)
            def progress_callback(step, total_steps):
                # Honor ComfyUI's interrupt button — raises
                # InterruptProcessingException, which propagates up out of
                # PiD's sampling loop and back to ComfyUI's executor.
                comfy.model_management.throw_exception_if_processing_interrupted()
                pbar.update_absolute(step + 1, total_steps, None)

            with torch.no_grad():
                samples = model.generate_samples_from_batch(
                    data_batch,
                    cfg_scale=cfg_scale,
                    num_steps=pid_inference_steps,
                    seed=seed,
                    shift=shift_val,
                    image_size=image_size,
                    callback=progress_callback,
                    sampler=sampler,
                    scheduler=scheduler,
                )
        else:
            import comfy.samplers
            from contextlib import nullcontext

            # Resolve default strings
            active_sampler = "euler" if sampler == "default" else sampler
            active_scheduler = "normal" if scheduler == "default" else scheduler

            # 1. Compute sigmas using ComfyUI's scheduler registry
            mock_sampling = MockModelSampling()
            sigmas = comfy.samplers.calculate_sigmas(mock_sampling, active_scheduler, pid_inference_steps)
            sigmas = sigmas.to(device)

            # 2. Apply shift if requested
            if shift_val is not None and shift_val > 0.0:
                sigmas = shift_val * sigmas / (1.0 + (shift_val - 1.0) * sigmas)
                sigmas = torch.clamp(sigmas, min=0.0, max=1.0)
                sigmas[-1] = 0.0

            # 3. Setup denoise function
            timescale = model.fm_trainer.timescale
            net = model.net
            autocast_ctx = torch.autocast(device.type, dtype=model.autocast_dtype) if (model.autocast_dtype and device.type != "cpu") else nullcontext()
            degrade_sigma_tensor = data_batch["degrade_sigma"]
            lq_video_or_image = data_batch["LQ_video_or_image"]
            lq_latent = data_batch["LQ_latent"]

            def denoise_fn(x_state, sigma_val):
                t_cur = sigma_val
                if not isinstance(t_cur, torch.Tensor):
                    t_cur = torch.tensor([float(t_cur)], device=x_state.device, dtype=x_state.dtype)
                elif t_cur.ndim == 0:
                    t_cur = t_cur.unsqueeze(0)
                
                B_current = x_state.shape[0]
                if t_cur.shape[0] != B_current:
                    t_cur = t_cur.expand(B_current)

                t_cur_scaled = t_cur * timescale

                # Run network forward pass
                with autocast_ctx:
                    v_pred = net(
                        x_state.to(dtype=model.autocast_dtype if model.autocast_dtype else torch.float32),
                        t_cur_scaled,
                        caption_embs,
                        lq_video_or_image=lq_video_or_image,
                        lq_latent=lq_latent,
                        degrade_sigma=degrade_sigma_tensor,
                    )
                
                # Convert velocity to x0 pred
                x0_pred = model._velocity_to_x0(x_state, v_pred, t_cur)
                return x0_pred.to(x_state.dtype)

            # 4. Prepare noise and generator
            gen = torch.Generator(device=device).manual_seed(seed)
            noise = torch.randn(B, 3, target_h, target_w, device=device, generator=gen)

            # 5. Instantiate Mock Model Wrap
            model_wrap = MockModelWrap(denoise_fn)

            # 6. Instantiate Sampler and Sample
            sampler_obj = comfy.samplers.sampler_object(active_sampler)
            
            pbar = comfy.utils.ProgressBar(pid_inference_steps)
            def comfy_callback(step, denoised, x_state, total_steps):
                comfy.model_management.throw_exception_if_processing_interrupted()
                pbar.update_absolute(step + 1, total_steps, None)

            with torch.no_grad():
                samples = sampler_obj.sample(
                    model_wrap,
                    sigmas,
                    extra_args={},
                    callback=comfy_callback,
                    noise=noise,
                )

        # ---- Convert PiD output to ComfyUI IMAGE format ----
        # PiD output: [B, 3, 1, H, W] in [-1, 1] (5D with T=1)
        # ComfyUI IMAGE: [B, H, W, 3] in [0, 1]
        output = samples.float().clamp(-1, 1)

        # Squeeze the temporal dimension if present
        if output.ndim == 5:
            output = output.squeeze(2)  # [B, 3, H, W]

        # Post-resize when match_original_size required PiD to run at its
        # training output resolution but the user asked for a different
        # VAE-native size. This is the second half of the "stay in
        # distribution then resize" strategy from the size-computation block
        # above; keeping the bilinear here (instead of inside the noise /
        # sampler loop) means PiD only sees in-distribution shapes.
        if need_post_resize and (output.shape[-2] != final_target_h or output.shape[-1] != final_target_w):
            import torch.nn.functional as F
            output = F.interpolate(
                output, size=(final_target_h, final_target_w),
                mode="bilinear", align_corners=False,
            )

        # Convert [-1, 1] → [0, 1] and rearrange to [B, H, W, C]
        output = (output + 1.0) / 2.0
        output = output.permute(0, 2, 3, 1).cpu()  # [B, H, W, 3]

        # Move model and its text encoder back to CPU (RAM) to free up VRAM
        model.to("cpu")
        if hasattr(model, "text_encoder") and model.text_encoder is not None:
            model.text_encoder.to("cpu")
        if hasattr(model, "_null_caption_embs") and isinstance(model._null_caption_embs, torch.Tensor):
            model._null_caption_embs = model._null_caption_embs.to("cpu")

        # Clear CUDA memory cache cleanly
        comfy.model_management.soft_empty_cache()

        return (output,)


# ---------------------------------------------------------------------------
# ComfyUI registration
# ---------------------------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "PiDDecode": PiDDecode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PiDDecode": "PiD Decode (Pixel Diffusion)",
}
