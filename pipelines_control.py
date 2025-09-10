# pipelines_sdxl_control.py
# SDXL ControlNet *Img2Img* that REUSES your existing SDXL base pipe (no 2× VRAM).

import torch
from PIL import Image
import numpy as np, cv2

from diffusers import StableDiffusionXLControlNetImg2ImgPipeline, ControlNetModel

# Cache so we don't rebuild the pipeline each call
_SDXL_CTRL_CACHE: dict[tuple[int, str], StableDiffusionXLControlNetImg2ImgPipeline] = {}

def _to8(x: int) -> int:
    return (x // 8) * 8

def _canny(img: Image.Image, low=100, high=200) -> Image.Image:
    arr = np.array(img.convert("RGB"))
    edges = cv2.Canny(arr, low, high)
    return Image.fromarray(cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB))

def get_sdxl_control_img2img_from_base(
    base_pipe,  # your existing StableDiffusionXLPipeline
    control_id: str = "diffusers/controlnet-canny-sdxl-1.0",
):
    """
    Build an SDXL ControlNet Img2Img pipeline by *reusing* the already-loaded SDXL base.
    We DO NOT call enable_model_cpu_offload() here to avoid Accelerate hook clashes.
    """
    if base_pipe is None:
        raise ValueError("SDXL base_pipe is None. Make sure init_sdxl() ran and `base` is set.")

    key = (id(base_pipe), control_id)
    if key in _SDXL_CTRL_CACHE:
        return _SDXL_CTRL_CACHE[key]

    # Match dtype/device to base
    try:
        dtype = base_pipe.unet.dtype
    except Exception:
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    controlnet = ControlNetModel.from_pretrained(control_id, torch_dtype=dtype)

    # Reuse components from base pipe
    pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_pipe(base_pipe, controlnet=controlnet)

    # >>> IMPORTANT: don't re-enable offload on the derived pipe (causes _hf_hook errors)
    # Move the small, new ControlNet parts to the same device as base
    try:
        base_device = next(base_pipe.unet.parameters()).device
    except Exception:
        base_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipe.to(base_device)

    # Light memory savers are safe
    try:
        pipe.vae.enable_slicing()
        pipe.vae.enable_tiling()
        pipe.enable_attention_slicing()
    except Exception:
        pass

    _SDXL_CTRL_CACHE[key] = pipe
    return pipe



def run_controlled_img2img_sdxl(
    base_pipe,                # StableDiffusionXLPipeline (already initialized)
    base_image: Image.Image,  # the init image to preserve (img2img)
    style_prompt: str,        # text defines the style
    negative_prompt: str | None = None,

    # sampling
    steps: int = 30,
    strength: float = 0.6,          # 0..1: how much to deviate from init image
    guidance_scale: float = 7.0,
    height: int | None = 1024,
    width: int | None = 1024,

    # control
    controlnet_conditioning_scale: float = 0.4,  # how strongly to follow control map
    control_guidance_start: float = 0.0,
    control_guidance_end: float = 1.0,
    canny_low: int = 100,
    canny_high: int = 200,

    # reproducibility
    seed: int | None = None,
):
    """
    SDXL ControlNet *Img2Img*:
      - `image=base_image` keeps the original photo's content
      - `control_image` enforces structure (e.g., Canny edges)
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

    pipe = get_sdxl_control_img2img_from_base(base_pipe)
    generator = torch.manual_seed(seed) if seed is not None else None

    out = pipe(
        prompt=style_prompt,
        negative_prompt=negative_prompt,
        image=init,                               # <-- Img2Img init image (preserves look)
        control_image=control_img,                # <-- ControlNet structural hint (edges/depth/pose)
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
