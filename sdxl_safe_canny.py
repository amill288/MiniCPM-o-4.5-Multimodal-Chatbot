import os
from typing import Optional, Tuple, Dict
import torch
import gradio as gr
from PIL import Image, ImageFilter

# Reduce fragmentation issues
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def _try_import_diffusers():
    try:
        from diffusers import (  # noqa: F401
            AutoPipelineForText2Image,
            AutoPipelineForImage2Image,
            ControlNetModel,
            StableDiffusionXLControlNetImg2ImgPipeline,
        )
        return True
    except Exception:
        return False


# -----------------------------
# Model choices
# -----------------------------
MODEL_CHOICES = {
    # SD 2.1 Turbo (non-SDXL). Very fast, lighter, lower fidelity.
    "turbo": "stabilityai/sd-turbo",
    # SDXL Turbo. Fast SDXL family.
    "xl-turbo": "stabilityai/sdxl-turbo",
    # SDXL Base 1.0 (normal). Higher quality, needs more steps, prefers CFG ~5-7.
    "normal": "stabilityai/stable-diffusion-xl-base-1.0",
}


def _is_turbo(model_key: str) -> bool:
    return model_key in ("turbo", "xl-turbo")


def _is_sdxl(model_key: str) -> bool:
    return model_key in ("xl-turbo", "normal")


# -----------------------------
# Refine ControlNet choices
# -----------------------------
REFINE_CONTROLNET_CHOICES = {
    "none": None,
    # SDXL canny controlnet
    "canny": "diffusers/controlnet-canny-sdxl-1.0",
}

# -----------------------------
# Pipelines (cached per model)
# -----------------------------
_T2I: Dict[str, object] = {}   # key: model_key -> pipeline
_I2I: Dict[str, object] = {}   # key: model_key -> pipeline
_I2I_REFINE_CN: Dict[Tuple[str, str], object] = {}  # key: (refine_model, control_key) -> pipeline
_CN: Dict[str, object] = {}    # key: control_key -> ControlNetModel


def _configure_pipe(pipe):
    # VRAM helpers (safe no-ops if unsupported)
    try:
        pipe.enable_attention_slicing()
    except Exception:
        pass
    try:
        pipe.enable_vae_slicing()
    except Exception:
        pass
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception:
        pass
    return pipe


def get_t2i(model_key: str):
    if model_key in _T2I:
        return _T2I[model_key]
    if not _try_import_diffusers():
        return None
    from diffusers import AutoPipelineForText2Image

    model_id = MODEL_CHOICES[model_key]
    variant = "fp16" if _is_sdxl(model_key) else None

    pipe = AutoPipelineForText2Image.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        variant=variant,
    )
    pipe = _configure_pipe(pipe).to("cpu")
    _T2I[model_key] = pipe
    return pipe


def get_i2i(model_key: str):
    if model_key in _I2I:
        return _I2I[model_key]
    if not _try_import_diffusers():
        return None
    from diffusers import AutoPipelineForImage2Image

    model_id = MODEL_CHOICES[model_key]
    variant = "fp16" if _is_sdxl(model_key) else None

    pipe = AutoPipelineForImage2Image.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        variant=variant,
    )
    pipe = _configure_pipe(pipe).to("cpu")
    _I2I[model_key] = pipe
    return pipe


def get_controlnet(control_key: str):
    if control_key == "none":
        return None
    if control_key in _CN:
        return _CN[control_key]
    if not _try_import_diffusers():
        return None
    from diffusers import ControlNetModel

    cn_id = REFINE_CONTROLNET_CHOICES.get(control_key)
    if cn_id is None:
        return None

    cn = ControlNetModel.from_pretrained(cn_id, torch_dtype=torch.float16)
    _CN[control_key] = cn
    return cn


def get_refine_i2i_with_controlnet(refine_model: str, control_key: str):
    """
    Only SDXL models support this SDXL ControlNet pipeline.
    For 'turbo' refine_model, we return None and will fall back to normal refine.
    """
    cache_key = (refine_model, control_key)
    if cache_key in _I2I_REFINE_CN:
        return _I2I_REFINE_CN[cache_key]
    if not _try_import_diffusers():
        return None
    if control_key == "none":
        return None
    if not _is_sdxl(refine_model):
        return None

    from diffusers import StableDiffusionXLControlNetImg2ImgPipeline

    model_id = MODEL_CHOICES[refine_model]
    cn = get_controlnet(control_key)
    if cn is None:
        return None

    pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
        model_id,
        controlnet=cn,
        torch_dtype=torch.float16,
        variant="fp16",
    )
    pipe = _configure_pipe(pipe).to("cpu")
    _I2I_REFINE_CN[cache_key] = pipe
    return pipe


def _to_cuda(pipe):
    if torch.cuda.is_available():
        pipe.to("cuda")


def _to_cpu_and_free(pipe):
    pipe.to("cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# -----------------------------
# Model-aware parameter normalization
# -----------------------------
def normalize_draft_params(model_key: str, steps: int, cfg: float, width: int, height: int):
    steps = int(steps)
    cfg = float(cfg)
    width = int(width)
    height = int(height)

    if _is_turbo(model_key):
        steps = max(1, min(8, steps))
        cfg = 0.0
        width = max(256, min(1024, width))
        height = max(256, min(1024, height))
        return steps, cfg, width, height

    # SDXL Base (normal)
    steps = max(35, min(500, steps))
    cfg = max(10.0, min(100.0, cfg))
    width = max(512, min(1024, width))
    height = max(512, min(1024, height))
    return steps, cfg, width, height


def normalize_refine_params(refine_model: str, draft_model: str, steps: int, strength: float, cfg: float):
    steps = int(steps)
    strength = float(strength)
    cfg = float(cfg)

    if _is_turbo(refine_model):
        steps = max(1, min(12, steps))
        cfg = 0.0
        strength = max(0.05, min(0.55, strength))
        return steps, strength, cfg

    steps = max(20, min(60, steps))
    cfg = max(3.0, min(8.0, cfg))

    if refine_model == "normal" and draft_model == "normal":
        strength = max(0.06, min(0.18, strength))  # polish-only
    else:
        strength = max(0.10, min(0.55, strength))

    return steps, strength, cfg


# -----------------------------
# Canny control image (refine)
# -----------------------------
def _ensure_rgb(im: Image.Image) -> Image.Image:
    return im.convert("RGB") if im.mode != "RGB" else im


def canny_like(im: Image.Image) -> Image.Image:
    # Lightweight fallback (no OpenCV). Good enough for "edge locking" in refine.
    edges = im.convert("L").filter(ImageFilter.FIND_EDGES)
    edges = edges.point(lambda p: 255 if p > 25 else 0)
    return edges.convert("RGB")


def build_refine_control_image(control_key: str, source_img: Image.Image) -> Optional[Image.Image]:
    if control_key == "none" or source_img is None:
        return None
    src = _ensure_rgb(source_img)
    # IMPORTANT: control image must match the exact size of the image being refined
    # (prevents tensor size mismatch errors)
    if control_key == "canny":
        return canny_like(src).resize(src.size, resample=Image.BICUBIC)
    return None


# -----------------------------
# Generation helpers
# -----------------------------
def generate_image(prompt: str, model_key: str, steps: int, width: int, height: int, cfg: float) -> Optional[Image.Image]:
    pipe = get_t2i(model_key)
    if pipe is None:
        return None

    steps, cfg, width, height = normalize_draft_params(model_key, steps, cfg, width, height)

    _to_cuda(pipe)
    with torch.inference_mode():
        img = pipe(
            prompt=prompt,
            num_inference_steps=steps,
            guidance_scale=cfg,
            height=height,
            width=width,
        ).images[0]
    _to_cpu_and_free(pipe)
    return img


def refine_image(
    img: Image.Image,
    prompt: str,
    refine_model: str,
    draft_model: str,
    steps: int,
    strength: float,
    cfg: float,
    refine_control_key: str,
    refine_control_scale: float,
) -> Tuple[Optional[Image.Image], Optional[Image.Image], str]:
    """
    Returns: (refined_img, control_preview_img, warning_text)
    """
    if img is None:
        return None, None, ""

    steps, strength, cfg = normalize_refine_params(refine_model, draft_model, steps, strength, cfg)

    # If user selected canny but refine_model isn't SDXL, we ignore CN.
    use_cn = (refine_control_key != "none") and _is_sdxl(refine_model)
    warn = ""
    ctrl_img = None

    if use_cn:
        ctrl_img = build_refine_control_image(refine_control_key, img)
        pipe_cn = get_refine_i2i_with_controlnet(refine_model, refine_control_key)
        if pipe_cn is None or ctrl_img is None:
            use_cn = False
            warn = "⚠️ Refine ControlNet requested but could not be loaded; falling back to normal refine."
    else:
        if refine_control_key != "none" and not _is_sdxl(refine_model):
            warn = "⚠️ Refine ControlNet (SDXL) is only supported for refine_model=xl-turbo/normal. Ignoring for turbo."

    if use_cn:
        _to_cuda(pipe_cn)
        with torch.inference_mode():
            out = pipe_cn(
                prompt=prompt,
                image=img,
                control_image=ctrl_img,
                controlnet_conditioning_scale=float(refine_control_scale),
                num_inference_steps=steps,
                strength=strength,
                guidance_scale=cfg,
            ).images[0]
        _to_cpu_and_free(pipe_cn)
        return out, ctrl_img, warn

    # Fallback: standard img2img refine
    pipe = get_i2i(refine_model)
    if pipe is None:
        return None, ctrl_img, "⚠️ Refine pipeline could not be loaded."
    _to_cuda(pipe)
    with torch.inference_mode():
        out = pipe(
            prompt=prompt,
            image=img,
            num_inference_steps=steps,
            strength=strength,
            guidance_scale=cfg,
        ).images[0]
    _to_cpu_and_free(pipe)
    return out, ctrl_img, warn


def do_generate_image(
    prompt: str,
    draft_model: str,
    refine_model: str,
    draft_steps: int,
    draft_cfg: float,
    width: int,
    height: int,
    do_refine: bool,
    refine_steps: int,
    refine_strength: float,
    refine_cfg: float,
    refine_control_key: str,
    refine_control_scale: float,
):
    prompt = (prompt or "").strip()
    if not prompt:
        return None, "", None

    img = generate_image(
        prompt=prompt,
        model_key=draft_model,
        steps=draft_steps,
        width=width,
        height=height,
        cfg=draft_cfg,
    )
    if img is None:
        return None, "❌ Draft generation failed (diffusers/model load).", None

    info_lines = []

    if do_refine:
        img2, ctrl_preview, warn = refine_image(
            img=img,
            prompt=prompt,
            refine_model=refine_model,
            draft_model=draft_model,
            steps=refine_steps,
            strength=refine_strength,
            cfg=refine_cfg,
            refine_control_key=refine_control_key,
            refine_control_scale=refine_control_scale,
        )
        if warn:
            info_lines.append(warn)
        img = img2 or img
    else:
        ctrl_preview = None

    info_lines.append(f"Draft model: **{draft_model}**")
    if do_refine:
        info_lines.append(f"Refine model: **{refine_model}** | ControlNet: **{refine_control_key}** (scale={float(refine_control_scale):.2f})")
    else:
        info_lines.append("Refine: **off**")

    return img, "\n\n".join(info_lines), ctrl_preview


# -----------------------------
# Gradio UI
# -----------------------------
with gr.Blocks(title="Stable Diffusion (Safe) + Refine Canny ControlNet") as demo:
    gr.Markdown(
        """
# Draft + optional Refine (with optional **Canny ControlNet** for refine)

This version adds a **Refine ControlNet** dropdown and a **Control preview** image.
- Refine ControlNet is implemented for **SDXL** refine models (`xl-turbo`, `normal`).
- If you select `turbo` as refine model, ControlNet will be ignored (with a warning).
        """
    )

    with gr.Row():
        img_prompt = gr.Textbox(label="Prompt", placeholder="A robot reading a book, cinematic lighting", lines=1)
        gen_img_btn = gr.Button("Generate image", variant="primary")

    with gr.Row():
        with gr.Column():
            draft_model = gr.Dropdown(list(MODEL_CHOICES.keys()), value="xl-turbo", label="Draft model (Turbo is 8 max)")
            draft_steps = gr.Slider(1, 500, value=8, step=1, label="Draft steps")
            draft_cfg = gr.Slider(0.0, 100.0, value=35.0, step=0.5, label="Draft CFG (Turbo forces 0)")
            sd_w = gr.Dropdown([384, 512, 640, 768, 1024], value=640, label="Width")
            sd_h = gr.Dropdown([384, 512, 640, 768, 1024], value=768, label="Height")

        with gr.Column():
            do_refine = gr.Checkbox(value=True, label="Refine (img2img)")
            refine_model = gr.Dropdown(list(MODEL_CHOICES.keys()), value="normal", label="Refine model")
            refine_steps = gr.Slider(1, 60, value=30, step=1, label="Refine steps")
            refine_strength = gr.Slider(0.05, 0.70, value=0.20, step=0.01, label="Refine strength")
            refine_cfg = gr.Slider(0.0, 9.0, value=3.0, step=0.5, label="Refine CFG (Turbo forces 0)")

            refine_control_key = gr.Dropdown(list(REFINE_CONTROLNET_CHOICES.keys()), value="canny", label="Refine ControlNet")
            refine_control_scale = gr.Slider(0.0, 2.0, value=1.0, step=0.05, label="Refine ControlNet scale")

    info_md = gr.Markdown(value="")
    with gr.Row():
        gen_img_out = gr.Image(label="Output")
        ctrl_preview = gr.Image(label="Refine Control preview (derived edges)", interactive=False)

    clear_btn_img = gr.Button("Clear", variant="stop")

    def _clear_img():
        return None, "", None, ""

    def _run(*args):
        img, info, ctrl = do_generate_image(*args)
        return img, ctrl, info

    # Generate
    img_prompt.submit(
        fn=_run,
        inputs=[
            img_prompt,
            draft_model, refine_model,
            draft_steps, draft_cfg,
            sd_w, sd_h,
            do_refine, refine_steps, refine_strength, refine_cfg,
            refine_control_key, refine_control_scale,
        ],
        outputs=[gen_img_out, ctrl_preview, info_md],
    )

    gen_img_btn.click(
        fn=_run,
        inputs=[
            img_prompt,
            draft_model, refine_model,
            draft_steps, draft_cfg,
            sd_w, sd_h,
            do_refine, refine_steps, refine_strength, refine_cfg,
            refine_control_key, refine_control_scale,
        ],
        outputs=[gen_img_out, ctrl_preview, info_md],
    )

    clear_btn_img.click(
        fn=_clear_img,
        inputs=[],
        outputs=[gen_img_out, img_prompt, ctrl_preview, info_md],
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
