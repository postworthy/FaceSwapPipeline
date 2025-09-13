# pipelines_control.py
# SDXL ControlNet Img2Img that REUSES your existing SDXL base pipe (no 2× VRAM)
import os
import json
from pathlib import Path

import torch
import numpy as np
import cv2
from PIL import Image, ImageOps
import piexif

from diffusers import StableDiffusionXLControlNetImg2ImgPipeline, ControlNetModel


# ---------- Internal helpers ----------
_SDXL_CTRL_CACHE: dict[tuple[int, str], StableDiffusionXLControlNetImg2ImgPipeline] = {}
_IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

def _to8(x: int) -> int:
    return (x // 8) * 8

def _canny(img: Image.Image, low: int = 100, high: int = 200) -> Image.Image:
    arr = np.array(img.convert("RGB"))
    edges = cv2.Canny(arr, low, high)
    return Image.fromarray(cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB))

def _preprocess_to_canvas(img: Image.Image, target_size: tuple[int, int]) -> Image.Image:
    """
    EXIF-aware transpose -> RGB -> contain into target_size -> paste on black canvas.
    Matches the snippet you provided.
    """
    img = ImageOps.exif_transpose(img).convert("RGB")
    fit = ImageOps.contain(img, target_size, method=Image.LANCZOS)
    canvas = Image.new("RGB", target_size, (0, 0, 0))
    x = (target_size[0] - fit.width) // 2
    y = (target_size[1] - fit.height) // 2
    canvas.paste(fit, (x, y))
    return canvas

def _iter_img_files(pathlike: str | os.PathLike | Path) -> list[Path]:
    p = Path(pathlike)
    if p.is_dir():
        return [f for f in sorted(p.iterdir()) if f.suffix.lower() in _IMG_EXTS]
    return [p] if p.suffix.lower() in _IMG_EXTS else []


# ---------- Pipeline builder (reuses base) ----------
def get_sdxl_control_img2img_from_base(
    base_pipe,  # existing StableDiffusionXLPipeline (already initialized in main.py)
    control_id: str = "diffusers/controlnet-canny-sdxl-1.0",
) -> StableDiffusionXLControlNetImg2ImgPipeline:
    """
    Build an SDXL ControlNet Img2Img pipeline by *reusing* the already-loaded SDXL base.
    We DO NOT call enable_model_cpu_offload() here to avoid Accelerate hook clashes.
    """
    if base_pipe is None:
        raise ValueError("SDXL base_pipe is None. Make sure init_sdxl() ran and `base` is set.")

    # Figure out the base device ONCE and use it consistently
    try:
        base_device = next(base_pipe.unet.parameters()).device
    except Exception:
        base_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _pin_pipe(pipe, dev: torch.device):
        # Move all known heavy bits and the controlnet(s), then sync execution device
        pipe.to(dev)
        for name in ("unet", "vae", "text_encoder", "text_encoder_2", "text_encoder_3", "transformer", "image_encoder"):
            m = getattr(pipe, name, None)
            if m is not None:
                try: m.to(dev)
                except Exception: pass
        cn = getattr(pipe, "controlnet", None)
        if cn is not None:
            items = cn if isinstance(cn, (list, tuple)) else [cn]
            for c in items:
                try: c.to(dev)
                except Exception: pass
        try:
            pipe._execution_device = dev
        except Exception:
            pass
        return pipe

    key = (id(base_pipe), control_id)
    if key in _SDXL_CTRL_CACHE:
        # IMPORTANT: if we previously cached it on another GPU, re-pin it now.
        return _pin_pipe(_SDXL_CTRL_CACHE[key], base_device)

    # Match dtype to base
    try:
        dtype = base_pipe.unet.dtype
    except Exception:
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    controlnet = ControlNetModel.from_pretrained(
        control_id,
        torch_dtype=dtype,
        use_safetensors=True,
        variant="fp16",
    )

    # Reuse components from base pipe
    pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_pipe(
        base_pipe,
        controlnet=controlnet,
        torch_dtype=dtype,
    )

    # Pin new pipe (and all submodules) to the same device as the base
    _pin_pipe(pipe, base_device)

    # Light memory savers are safe
    try:
        pipe.vae.enable_slicing()
        pipe.vae.enable_tiling()
        pipe.enable_attention_slicing()
    except Exception:
        pass

    _SDXL_CTRL_CACHE[key] = pipe
    return pipe



# ---------- Single-image runner (used internally) ----------
def run_controlled_img2img_sdxl(
    base_pipe,                # StableDiffusionXLPipeline (already initialized)
    base_image: Image.Image,  # init image to preserve (img2img)
    style_prompt: str,        # text defines the style
    negative_prompt: str | None = None,

    # sampling
    steps: int = 30,
    strength: float = 0.6,          # 0..1: how much to deviate from init image
    guidance_scale: float = 7.0,
    height: int | None = 1024,
    width: int | None = 1024,

    # control
    controlnet_conditioning_scale: float = 0.8,  # follow control map strength
    control_guidance_start: float = 0.0,
    control_guidance_end: float = 1.0,
    canny_low: int = 100,
    canny_high: int = 200,

    # reproducibility
    seed: int | None = None,

    # which ControlNet
    control_id: str = "diffusers/controlnet-canny-sdxl-1.0",
):
    """
    SDXL ControlNet *Img2Img*:
      - `image=base_image` preserves source content
      - `control_image` enforces structure (edges/depth/pose)
      - `style_prompt` imposes the look

    Returns: list[PIL.Image] (len=1)
    """
    # Normalize dims to multiples of 8
    if height is None: height = 1024
    if width  is None: width  = 1024
    height, width = _to8(height), _to8(width)

    # Prepare init and control images
    init = base_image.convert("RGB").resize((width, height), Image.LANCZOS)
    control_img = _canny(init, canny_low, canny_high)

    pipe = get_sdxl_control_img2img_from_base(base_pipe, control_id=control_id)
    dev = getattr(pipe, "_execution_device", next(pipe.unet.parameters()).device)
    generator = torch.Generator(device=str(dev)).manual_seed(seed) if seed is not None else None


    out = pipe(
        prompt=style_prompt,
        negative_prompt=negative_prompt,
        image=init,                               # Img2Img init image (preserves look)
        control_image=control_img,                # ControlNet structural hint (edges)
        controlnet_conditioning_scale=controlnet_conditioning_scale,
        control_guidance_start=control_guidance_start,
        control_guidance_end=control_guidance_end,
        num_inference_steps=steps,
        strength=strength,                        # higher = more stylization / less init preservation
        guidance_scale=guidance_scale,
        height=height,
        width=width,
        generator=generator,
    )
    return out.images


# ---------- Public API (mimics run_sdxl’s seed & file-naming) ----------
def run_control(
    base_pipe,                        # pass _ensure_sdxl() from main
    input_path_or_image,              # str|Path (file or directory) OR PIL.Image.Image
    style_prompt: str | None = None,
    negative_prompt: str | None = None,

    # sampling / img2img
    steps: int = 30,
    strength: float = 0.8,
    guidance_scale: float = 7.0,
    height: int | None = 1024,
    width: int | None = 1024,

    # control
    controlnet_conditioning_scale: float = 0.5,
    control_guidance_start: float = 0.0,
    control_guidance_end: float = 1.0,
    canny_low: int = 100,
    canny_high: int = 200,
    control_id: str = "diffusers/controlnet-canny-sdxl-1.0",

    # reproducibility (MIMIC run_sdxl)
    seed: int = 0,

    # optional post-process (like tune_with_func in run_sdxl)
    tune_with_func=None,
    reverse_swap_colors: bool = False,

    # IO (MIMIC run_sdxl naming)
    output_dir: str = "/app/output",
    save_output: bool = True,
) -> dict[str, Image.Image]:
    """
    Polymorphic runner:
      - If input_path_or_image is a path (file or directory), process all matching images.
      - If it's a PIL.Image, process just that image.

    Preprocess for each image:
      - EXIF transpose, RGB, aspect-ratio preserve with `ImageOps.contain`,
        paste to black canvas of size (width,height).

    Returns dict[path -> PIL.Image] like your other runners.
    """
    print(f"Seed: {seed}")
    # Normalize dims to multiples of 8
    if height is None: height = 1024
    if width  is None: width  = 1024
    height, width = _to8(height), _to8(width)

    os.makedirs(output_dir, exist_ok=True)
    results: dict[str, Image.Image] = {}

    # Prepare a shared generator (mimic run_sdxl behavior)
    generator = torch.manual_seed(seed)

    # Counter to mimic your /app/output/output-{seed}-{i}.png naming
    out_idx = 0

    # Build a reusable exif writer
    def _save_with_exif(pil_img: Image.Image, file_path: str, used_seed: int):
        exif_comment = json.dumps({
            "Model": "stabilityai/stable-diffusion-xl-base-1.0",
            "ControlNet": control_id,
            "Seed": str(used_seed),
            "Steps": str(steps),
            "Prompt": style_prompt,
            "Negative Prompt": negative_prompt if negative_prompt is not None else "",
            "Strength": str(strength),
            "Guidance Scale": str(guidance_scale),
            "ControlNet Conditioning Scale": str(controlnet_conditioning_scale),
            "Height": str(height),
            "Width": str(width),
            "Face Swapped": "True" if callable(tune_with_func) else "False",
        })
        comment_bytes = exif_comment.encode("utf-8")
        exif_dict = {"Exif": {}}
        exif_dict["Exif"][piexif.ExifIFD.UserComment] = comment_bytes
        exif_bytes = piexif.dump(exif_dict)
        if save_output:
            pil_img.save(file_path, exif=exif_bytes)
        results[file_path] = pil_img

    def _run_one(pil_img: Image.Image):
        nonlocal out_idx

        # Preprocess onto black canvas at target WxH
        init_img = _preprocess_to_canvas(pil_img, (width, height))

        # Run the actual SDXL ControlNet Img2Img with our shared generator
        imgs = run_controlled_img2img_sdxl(
            base_pipe=base_pipe,
            base_image=init_img,
            style_prompt=style_prompt,
            negative_prompt=negative_prompt,
            steps=steps,
            strength=strength,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            control_guidance_start=control_guidance_start,
            control_guidance_end=control_guidance_end,
            canny_low=canny_low,
            canny_high=canny_high,
            seed=seed,                 # we pass the same seed here
            control_id=control_id,
        )

        # imgs is a list (usually len=1)
        for image in imgs:
            if callable(tune_with_func):
                # maintain parity with your run_sdxl tune behavior
                if not reverse_swap_colors:
                    modified_list = list(tune_with_func(image))
                else:
                    arr = np.array(image)
                    modified_list = list(tune_with_func(arr[:, :, ::-1]))
                for mod_i, modified_image in enumerate(modified_list):
                    file_name = os.path.join(output_dir, f"output-{seed}-{out_idx}.png")
                    _save_with_exif(modified_image, file_name, seed)
                    out_idx += 1
            else:
                file_name = os.path.join(output_dir, f"output-{seed}-{out_idx}.png")
                _save_with_exif(image, file_name, seed)
                out_idx += 1

    # Branch on input type
    if isinstance(input_path_or_image, (str, os.PathLike, Path)):
        files = _iter_img_files(input_path_or_image)
        if not files:
            print(f"No images found at: {input_path_or_image}")
            return {}

        for f in files:
            try:
                raw = Image.open(str(f))
            except Exception as e:
                print(f"Skipping {f.name}: {e}")
                continue
            _run_one(raw)

    elif isinstance(input_path_or_image, Image.Image):
        _run_one(input_path_or_image)
    else:
        raise TypeError("run_control expects a path (file/dir) or a PIL.Image.Image")

    return results
