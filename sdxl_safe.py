import os
from typing import Optional
import torch
import gradio as gr
from PIL import Image

# Reduce fragmentation issues
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
    # SDXL Base 1.0 (normal). Higher quality, needs more steps, prefers CFG ~5-7.
    "normal": "stabilityai/stable-diffusion-xl-base-1.0",
}


def _is_turbo(model_key: str) -> bool:
    return model_key in ("turbo", "xl-turbo")


def _is_sdxl(model_key: str) -> bool:
    return model_key in ("xl-turbo", "normal")


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
    # SDXL repos provide fp16 variants; sd-turbo may not.
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
    """
    Make "sane defaults" per model. This is the big fix:
    SDXL base (normal) looks *broken* if you drive it like Turbo (few steps / low CFG),
    especially at 512px. We force realistic minimums.
    """
    steps = int(steps)
    cfg = float(cfg)
    width = int(width)
    height = int(height)

    if _is_turbo(model_key):
        # Turbo expects very few steps and effectively CFG=0.
        steps = max(1, min(8, steps))
        cfg = 0.0
        # Turbo works well at 512; larger is okay but more VRAM.
        width = max(256, min(1024, width))
        height = max(256, min(1024, height))
        return steps, cfg, width, height

    # ---- Normal SDXL Base ----
    # SDXL base is trained around 1024px and needs more steps.
    # Driving it with 10 steps + CFG=1 often creates "mosaic / stained glass" nonsense.
    steps = max(35.0, min(500, steps))
    cfg = max(10.0, min(100,cfg))

    # If user selected 512, allow it, but SDXL base generally looks much better at 768/1024.
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
        strength = max(0.05, min(0.45, strength))
        return steps, strength, cfg

    # Normal SDXL refine
    steps = max(20, steps)
    cfg = max(3.0, min(8.0, cfg))

    # If it's normal->normal, keep strength very low (polish only) to avoid pattern hallucination.
    if refine_model == "normal" and draft_model == "normal":
        strength = max(0.06, min(0.18, strength))
    else:
        # turbo->normal can tolerate a bit more change
        strength = max(0.10, min(0.45, strength))

    return steps, strength, cfg


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
) -> Optional[Image.Image]:
    pipe = get_i2i(refine_model)
    if pipe is None or img is None:
        return None

    steps, strength, cfg = normalize_refine_params(refine_model, draft_model, steps, strength, cfg)

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
            refine_model=refine_model,
            draft_model=draft_model,
            steps=refine_steps,
            strength=refine_strength,
            cfg=refine_cfg,
        )
        return img2 or img

    return img


# -----------------------------
# UI helpers: show effective settings
# -----------------------------
def explain_effective_settings(draft_model, refine_model, draft_steps, draft_cfg, w, h, do_refine, refine_steps, refine_strength, refine_cfg):
    ds, dcfg, ww, hh = normalize_draft_params(draft_model, draft_steps, draft_cfg, w, h)
    lines = [
        f"**Effective Draft:** model={draft_model} | steps={ds} | cfg={dcfg:.1f} | size={ww}×{hh}"
    ]
    if do_refine:
        rs, rstr, rcfg = normalize_refine_params(refine_model, draft_model, refine_steps, refine_strength, refine_cfg)
        lines.append(f"**Effective Refine:** model={refine_model} | steps={rs} | strength={rstr:.2f} | cfg={rcfg:.1f}")
        if refine_model == "normal" and draft_model == "normal":
            lines.append("ℹ️ Normal→Normal: strength is clamped low to prevent repeating/mosaic artifacts.")
    else:
        lines.append("**Refine:** off")
    if draft_model == "normal" and (w < 768 or h < 768):
        lines.append("⚠️ SDXL Base looks best at 768/1024. 512 can look unstable with some prompts.")
    return "\n\n".join(lines)


# -----------------------------
# Gradio UI
# -----------------------------
with gr.Blocks(title="Stable Diffusion (turbo / xl-turbo / normal) - Draft + Refine") as demo:
    gr.Markdown(
        """
# Draft + optional HD refine (img2img)

This app supports **three model families** for **both** Draft and Refine:
- **turbo** = `stabilityai/sd-turbo` (fast, non-SDXL, CFG forced to 0)
- **xl-turbo** = `stabilityai/sdxl-turbo` (fast SDXL, CFG forced to 0)
- **normal** = `stabilityai/stable-diffusion-xl-base-1.0` (quality SDXL)

**Important:** The same slider values do **not** mean the same thing across models.
To avoid the “mosaic / stained-glass” failures you saw, the app applies **model-aware clamps**:
- Turbo variants: low steps, **CFG=0**
- SDXL Base (normal): **>=25 steps**, **CFG ~5–7**
- Normal→Normal refine: **very low strength**
        """
    )

    with gr.Row():
        img_prompt = gr.Textbox(label="Prompt", placeholder="A robot reading a book, cinematic lighting", lines=1)
        gen_img_btn = gr.Button("Generate image", variant="primary")

    with gr.Row():
        with gr.Column():
            draft_model = gr.Dropdown(list(MODEL_CHOICES.keys()), value="normal", label="Draft model")
            draft_steps = gr.Slider(1, 500, value=50, step=1, label="Draft steps")
            draft_cfg = gr.Slider(0.0, 100.0, value=35.0, step=0.5, label="Draft CFG (Turbo forces 0)")
            sd_w = gr.Dropdown([384, 512, 640, 768, 1024], value=768, label="Width")
            sd_h = gr.Dropdown([384, 512, 640, 768, 1024], value=1024, label="Height")

        with gr.Column():
            do_refine = gr.Checkbox(value=False, label="HD refine (img2img)")
            refine_model = gr.Dropdown(list(MODEL_CHOICES.keys()), value="normal", label="Refine model")
            refine_steps = gr.Slider(1, 500, value=30, step=1, label="Refine steps")
            refine_strength = gr.Slider(0.05, 0.70, value=0.20, step=0.01, label="Refine strength")
            refine_cfg = gr.Slider(0.0, 50.0, value=3.0, step=0.5, label="Refine CFG (Turbo forces 0)")

    effective_md = gr.Markdown(value="")

    with gr.Row():
        clear_btn_img = gr.Button("Clear", variant="stop")

    gen_img_out = gr.Image(label="Output")

    def _clear_img():
        return None, "", ""

    # Update "effective settings" readout whenever controls change
    for c in [draft_model, refine_model, draft_steps, draft_cfg, sd_w, sd_h, do_refine, refine_steps, refine_strength, refine_cfg]:
        c.change(
            fn=explain_effective_settings,
            inputs=[draft_model, refine_model, draft_steps, draft_cfg, sd_w, sd_h, do_refine, refine_steps, refine_strength, refine_cfg],
            outputs=[effective_md],
        )

    # Generate (submit and button)
    img_prompt.submit(
        fn=do_generate_image,
        inputs=[img_prompt, draft_model, refine_model, draft_steps, draft_cfg, sd_w, sd_h, do_refine, refine_steps, refine_strength, refine_cfg],
        outputs=[gen_img_out],
    )
    gen_img_btn.click(
        fn=do_generate_image,
        inputs=[img_prompt, draft_model, refine_model, draft_steps, draft_cfg, sd_w, sd_h, do_refine, refine_steps, refine_strength, refine_cfg],
        outputs=[gen_img_out],
    )

    clear_btn_img.click(
        fn=_clear_img,
        inputs=[],
        outputs=[gen_img_out, img_prompt, effective_md],
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
