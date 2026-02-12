import os
import time
import tempfile
from typing import Optional
import torch
import gradio as gr
from PIL import Image

# Keep history bounded to avoid VRAM/latency blowups
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def _try_import_diffusers():
    try:
        from diffusers import AutoPipelineForText2Image, AutoPipelineForImage2Image  # noqa: F401
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
    # SDXL Base 1.0 (normal). Higher quality, needs more steps, prefers nonzero CFG.
    "normal": "stabilityai/stable-diffusion-xl-base-1.0",
}


def _is_turbo(model_key: str) -> bool:
    return model_key in ("turbo", "xl-turbo")


# -----------------------------
# Pipelines (cached per model)
# -----------------------------
_T2I = {}   # key: model_key -> pipeline
_I2I = {}   # key: model_key -> pipeline


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
    # xformers (optional)
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception:
        pass
    return pipe


def get_t2i(model_key: str):
    global _T2I
    if model_key in _T2I:
        return _T2I[model_key]
    if not _try_import_diffusers():
        return None
    from diffusers import AutoPipelineForText2Image

    model_id = MODEL_CHOICES[model_key]
    pipe = AutoPipelineForText2Image.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        variant="fp16" if "xl" in model_id or "stable-diffusion-xl" in model_id else None,
    )
    pipe = _configure_pipe(pipe).to("cpu")
    _T2I[model_key] = pipe
    return pipe


def get_i2i(model_key: str):
    global _I2I
    if model_key in _I2I:
        return _I2I[model_key]
    if not _try_import_diffusers():
        return None
    from diffusers import AutoPipelineForImage2Image

    model_id = MODEL_CHOICES[model_key]
    pipe = AutoPipelineForImage2Image.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        variant="fp16" if "xl" in model_id or "stable-diffusion-xl" in model_id else None,
    )
    pipe = _configure_pipe(pipe).to("cpu")
    _I2I[model_key] = pipe
    return pipe


def _to_cuda(pipe):
    if torch.cuda.is_available():
        pipe.to("cuda")


def _to_cpu_and_free(pipe):
    pipe.to("cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# -----------------------------
# Generation helpers
# -----------------------------
def generate_image(
    prompt: str,
    model_key: str,
    steps: int,
    width: int,
    height: int,
    cfg: float,
) -> Optional[Image.Image]:
    pipe = get_t2i(model_key)
    if pipe is None:
        return None

    # Turbo models are trained for very few steps and effectively expect CFG ~ 0.
    if _is_turbo(model_key):
        steps = int(max(1, steps))
        cfg = 0.0
    else:
        # Normal SDXL wants more steps + nonzero CFG.
        steps = int(max(10, steps))
        cfg = float(max(1.0, cfg))

    _to_cuda(pipe)
    with torch.inference_mode():
        img = pipe(
            prompt=prompt,
            num_inference_steps=steps,
            guidance_scale=cfg,
            height=int(height),
            width=int(width),
        ).images[0]
    _to_cpu_and_free(pipe)
    return img


def refine_image(
    img: Image.Image,
    prompt: str,
    model_key: str,
    steps: int,
    strength: float,
    cfg: float,
    draft_model_key: str,
) -> Optional[Image.Image]:
    pipe = get_i2i(model_key)
    if pipe is None or img is None:
        return None

    # --- Step/CFG/strength semantics differ by model ---
    if _is_turbo(model_key):
        # Turbo img2img is touchy; keep it light.
        steps = int(max(1, steps))
        cfg = 0.0
        strength = float(max(0.05, min(0.45, strength)))
    else:
        # Normal SDXL img2img: best results are usually with modest strength.
        steps = int(max(10, steps))
        cfg = float(max(1.0, cfg))

        # If you're doing normal->normal (draft=normal and refine=normal), keep strength *very* low.
        if (draft_model_key == "normal") and (model_key == "normal"):
            strength = float(max(0.06, strength))
        else:
            # turbo -> normal refinement can tolerate a bit more
            strength = float(max(0.10, strength))

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
    return out


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
):
    prompt = (prompt or "").strip()
    if not prompt:
        return None

    img = generate_image(
        prompt=prompt,
        model_key=draft_model,
        steps=draft_steps,
        width=width,
        height=height,
        cfg=draft_cfg,
    )
    if img is None:
        return None

    if do_refine:
        img2 = refine_image(
            img=img,
            prompt=prompt,
            model_key=refine_model,
            steps=refine_steps,
            strength=refine_strength,
            cfg=refine_cfg,
            draft_model_key=draft_model,
        )
        return img2 or img

    return img


# -----------------------------
# Gradio UI
# -----------------------------
with gr.Blocks(title="Stable Diffusion (Turbo / XL-Turbo / Normal) - Draft + Refine") as demo:
    gr.Markdown(
        """
# Image generation (Draft + optional HD refine)

**Draft model** and **Refine model** can be chosen independently:
- **turbo** = `stabilityai/sd-turbo` (fast, non-SDXL)
- **xl-turbo** = `stabilityai/sdxl-turbo` (fast SDXL)
- **normal** = `stabilityai/stable-diffusion-xl-base-1.0` (quality; needs more steps & nonzero CFG)

**Note:** Slider meaning differs by model. This app automatically clamps/overrides unsafe settings:
- Turbo variants force **CFG=0** and clamp steps low.
- Normal SDXL expects **more steps** and **CFG > 0**.
- Normal→Normal refinement clamps strength very low to prevent the “stained-glass / repeating pattern” effect.
        """
    )

    with gr.Row():
        img_prompt = gr.Textbox(label="Image prompt", placeholder="A robot reading a book, cinematic lighting", lines=1)
        gen_img_btn = gr.Button("Generate image", variant="primary")

    with gr.Row():
        with gr.Column():
            draft_model = gr.Dropdown(
                choices=list(MODEL_CHOICES.keys()),
                value="xl-turbo",
                label="Draft model",
            )
            draft_steps = gr.Slider(1, 2000, value=50, step=1, label="Draft steps")
            draft_cfg = gr.Slider(0.0, 20.0, value=0.0, step=0.5, label="Draft CFG (ignored for Turbo)")
            sd_w = gr.Dropdown([384, 512, 640, 768, 1024], value=512, label="Width")
            sd_h = gr.Dropdown([384, 512, 640, 768, 1024], value=512, label="Height")

        with gr.Column():
            do_refine = gr.Checkbox(value=True, label="HD refine (img2img)")
            refine_model = gr.Dropdown(
                choices=list(MODEL_CHOICES.keys()),
                value="normal",
                label="Refine model",
            )
            refine_steps = gr.Slider(1, 1000, value=30, step=1, label="Refine steps")
            refine_strength = gr.Slider(0.05, 0.70, value=0.20, step=0.01, label="Refine strength")
            refine_cfg = gr.Slider(0.0, 20.0, value=3.0, step=0.5, label="Refine CFG (ignored for Turbo)")

    with gr.Row():
        clear_btn_img = gr.Button("Clear", variant="stop")
    gen_img_out = gr.Image(label="Generated image")

    def _clear_img():
        return None, ""

    # Generate
    img_prompt.submit(
        fn=do_generate_image,
        inputs=[
            img_prompt,
            draft_model, refine_model,
            draft_steps, draft_cfg,
            sd_w, sd_h,
            do_refine, refine_steps, refine_strength, refine_cfg
        ],
        outputs=[gen_img_out],
    )

    gen_img_btn.click(
        fn=do_generate_image,
        inputs=[
            img_prompt,
            draft_model, refine_model,
            draft_steps, draft_cfg,
            sd_w, sd_h,
            do_refine, refine_steps, refine_strength, refine_cfg
        ],
        outputs=[gen_img_out],
    )

    clear_btn_img.click(
        fn=_clear_img,
        inputs=[],
        outputs=[gen_img_out, img_prompt],
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
