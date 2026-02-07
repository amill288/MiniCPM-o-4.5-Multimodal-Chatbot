import os
import time
import tempfile
from typing import List, Dict, Any, Optional, Tuple
import torch
import gradio as gr
from PIL import Image
import sys
import traceback
import subprocess, re
from transformers import AutoModel, AutoTokenizer
import numpy as np


# Keep history bounded to avoid VRAM/latency blowups
MAX_TURNS = 12

# If you previously saw fragmentation warnings, this helps:
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# -----------------------------
# Lazy imports (optional deps)
# -----------------------------
def _try_import_faster_whisper():
    try:
        from faster_whisper import WhisperModel  # noqa: F401
        return True
    except Exception:
        return False


def _try_import_diffusers():
    try:
        from diffusers import AutoPipelineForText2Image  # noqa: F401
        return True
    except Exception:
        return False


_MODEL = None
_TOKENIZER = None

def get_minicpm_model():
    global _MODEL
    if _MODEL is not None:
        return _MODEL

    model_id = "openbmb/MiniCPM-V-4_5-int4"
    _MODEL = AutoModel.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto",   # IMPORTANT
    ).eval()
    return _MODEL

def get_minicpm_tokenizer():
    global _TOKENIZER
    if _TOKENIZER is not None:
        return _TOKENIZER

    model_id = "openbmb/MiniCPM-V-4_5-int4"
    _TOKENIZER = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
    )
    return _TOKENIZER



# -----------------------------
# Voice: Speech-to-Text (CPU)
# -----------------------------
_WHISPER = None

def transcribe_audio(audio_path: str) -> str:
    """
    Uses faster-whisper (CPU by default). Keeps GPU free for MiniCPM.
    """
    global _WHISPER
    if not _try_import_faster_whisper():
        return "ERROR: faster-whisper not installed. `pip install faster-whisper`"

    from faster_whisper import WhisperModel

    if _WHISPER is None:
        # "small" is a good speed/quality tradeoff on CPU
        _WHISPER = WhisperModel("small", device="cpu", compute_type="int8")

    segments, info = _WHISPER.transcribe(audio_path, beam_size=5, vad_filter=True)
    text = "".join(seg.text for seg in segments).strip()
    if not text:
        return "(no speech detected)"
    return text



# -----------------------------
# Voice: Text-to-Speech (CPU)
# -----------------------------

# Path to the LibriTTS voice you downloaded
PIPER_VOICE = os.path.expanduser("~/piper_voices/libritts_r_medium/en_US-libritts_r-medium.onnx")

def tts_to_wav(text: str) -> str:
    """
    Generate speech audio (offline) using Piper voice.
    """
    out_wav = os.path.join(tempfile.gettempdir(), f"minicpm_tts_{int(time.time()*1000)}.wav")
    
    cmd = [
        "piper",
        "--model", PIPER_VOICE,
        "--output_file", out_wav,
    ]
    
    # Piper takes input text from stdin
    subprocess.run(cmd, input=text.encode("utf-8"), check=True)
    
    return out_wav


def clean_for_tts(text: str) -> str:
    # remove markdown emphasis
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"\*(.*?)\*", r"\1", text)

    # normalize bullets
    text = re.sub(r"^\s*[-•*]\s+", "", text, flags=re.MULTILINE)

    return text.strip()

# -----------------------------
# OPTIONAL: Image generation (separate model)
# -----------------------------
_SD = None

def get_sd_pipeline():
    """
    Optional image generator. WARNING: running SD alongside the LLM can be tight on 12GB.
    We load it on CPU and move to GPU only for generation, then move back.
    """
    global _SD
    if _SD is not None:
        return _SD
    if not _try_import_diffusers():
        return None

    from diffusers import AutoPipelineForText2Image

    # A fast/small-ish option. You can change this to another SD checkpoint.
    sd_id = "stabilityai/sd-turbo"

    pipe = AutoPipelineForText2Image.from_pretrained(
        sd_id,
        torch_dtype=torch.float16,
        variant="fp16",
    )
    pipe = pipe.to("cpu")
    _SD = pipe
    return pipe


def generate_image(prompt: str, steps: int = 4) -> Optional[Image.Image]:
    pipe = get_sd_pipeline()
    if pipe is None:
        return None

    # Move to GPU briefly
    if torch.cuda.is_available():
        pipe.to("cuda")

    with torch.inference_mode():
        img = pipe(prompt=prompt, num_inference_steps=steps, guidance_scale=0.0).images[0]

    # Move back to CPU and free VRAM
    pipe.to("cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return img





def _unique_wav_path(prefix="tts"):
    return os.path.join("/tmp", f"{prefix}_{int(time.time()*1000)}.wav")

def wav_apply_volume(in_wav: str, volume: float) -> str:
    """
    Returns a new WAV with audio scaled by `volume`.
    volume=1.0 unchanged, 0.5 half as loud, 2.0 twice as loud (may clip).
    """
    import soundfile as sf

    volume = float(volume)
    if volume <= 0:
        volume = 0.0001

    audio, sr = sf.read(in_wav, always_2d=False)  # audio shape: (n,) or (n, ch)

    # Scale
    audio = audio * volume

    # Prevent clipping: clamp to [-1, 1] for float WAVs
    # soundfile may give float32/float64; clamp is safe either way.
    audio = np.clip(audio, -1.0, 1.0)

    out_wav = _unique_wav_path("tts_vol")
    sf.write(out_wav, audio, sr)
    return out_wav

def wav_reverse(in_wav: str) -> str:
    """
    Returns a new WAV reversed in time.
    """
    import soundfile as sf

    audio, sr = sf.read(in_wav, always_2d=False)

    # Reverse along time axis
    if audio.ndim == 1:
        audio_rev = audio[::-1]
    else:
        audio_rev = audio[::-1, :]

    out_wav = _unique_wav_path("tts_rev")
    sf.write(out_wav, audio_rev, sr)
    return out_wav






# -----------------------------
# Chat function 
# -----------------------------
def chat_step(chat_ui, msgs_state, user_text, user_image, speak_back, tts_volume_val,reverse_after, last_audio_state):

    # normalize inputs (IMPORTANT)
    tts_volume_val = 1.0 if tts_volume_val is None else float(tts_volume_val)
    reverse_after = bool(reverse_after) if reverse_after is not None else False

    # -------- Chat UI normalization (Gradio Chatbot) --------
    def _as_messages(x):
        if x is None:
            return []
        # New "messages" format: list[dict]
        if isinstance(x, list) and (len(x) == 0 or isinstance(x[0], dict)):
            return x
        # Legacy format: list[(user, assistant)]
        if isinstance(x, list) and len(x) > 0 and isinstance(x[0], (tuple, list)) and len(x[0]) == 2:
            msgs = []
            for u, a in x:
                msgs.append({"role": "user", "content": str(u)})
                msgs.append({"role": "assistant", "content": str(a)})
            return msgs
        return []

    def _trim_history_safe(msgs, max_turns=6):
        # keep last N user turns (+ their assistant replies)
        msgs = msgs or []
        user_idxs = [i for i, m in enumerate(msgs) if m.get("role") == "user"]
        if len(user_idxs) <= max_turns:
            return msgs
        cut = user_idxs[-max_turns]
        return msgs[cut:]

    # -------- Input cleanup --------
    user_text = (user_text or "").strip()
    if not user_text and user_image is None:
        return _as_messages(chat_ui), (msgs_state or []), "", None

    # Normalize image coming from Gradio (can be PIL or np.ndarray)
    pil_img = None
    if user_image is not None:
        if isinstance(user_image, np.ndarray):
            pil_img = Image.fromarray(user_image.astype(np.uint8))
        elif isinstance(user_image, Image.Image):
            pil_img = user_image
        else:
            # Unknown type (rare) → just fail gracefully
            chat_ui_msgs = _as_messages(chat_ui)
            chat_ui_msgs.append({"role": "user", "content": user_text or "[image]"})
            chat_ui_msgs.append({"role": "assistant", "content": "⚠️ Unsupported image type received from UI."})
            return chat_ui_msgs, (msgs_state or []), "", None

        pil_img = pil_img.convert("RGB")

    # -------- Load model + tokenizer (must be AutoTokenizer, not AutoProcessor) --------
    model = get_minicpm_model()
    tokenizer = get_minicpm_tokenizer()

    # -------- Build MiniCPM message (THIS is the correct format) --------
    # UI should only store text, not raw PIL
    ui_user_content = user_text if user_text else ("[image]" if pil_img is not None else "")

    if pil_img is not None:
        mm_content = [pil_img, user_text if user_text else "Describe this image."]
    else:
        mm_content = user_text

    msgs_state = msgs_state or []
    msgs_state.append({"role": "user", "content": mm_content})
    msgs_state = _trim_history_safe(msgs_state, max_turns=6)


    try:
        # IMPORTANT: pass tokenizer=tokenizer exactly like the HF README
        response_text = model.chat(
            msgs=msgs_state,
            tokenizer=tokenizer,
            stream=False,
            enable_thinking=False,
            use_tts_template=False,
            # These help stability for images:
            max_slice_nums=1,
            use_image_id=False,
        )

        # Update model state
        msgs_state.append({"role": "assistant", "content": response_text})
        msgs_state = _trim_history_safe(msgs_state, max_turns=6)

        # Update UI chat (messages format)
        chat_ui_msgs = _as_messages(chat_ui)
        chat_ui_msgs.append({"role": "user", "content": ui_user_content})
        chat_ui_msgs.append({"role": "assistant", "content": response_text})

        audio_path = None
        new_last_audio = last_audio_state
        new_is_reversed = False  # any new spoken reply resets to normal

        if speak_back:
            normal_path = tts_to_wav(clean_for_tts(response_text))
            normal_path = wav_apply_volume(normal_path, tts_volume_val)

            audio_path = normal_path  # default playback is normal
            new_last_audio = normal_path
            new_is_reversed = False

            # OPTIONAL: if you still want "reverse_after" checkbox to play reversed immediately:
            if reverse_after:
                audio_path = wav_reverse(normal_path)
                new_is_reversed = True

        return chat_ui_msgs, msgs_state, "", audio_path, new_last_audio, new_is_reversed

    


    except torch.OutOfMemoryError:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # rollback last user message
        if msgs_state and msgs_state[-1].get("role") == "user":
            msgs_state.pop()

        chat_ui_msgs = _as_messages(chat_ui)
        chat_ui_msgs.append({"role": "user", "content": ui_user_content})
        chat_ui_msgs.append({"role": "assistant", "content": "⚠️ GPU OOM. Try a smaller image or shorter prompt."})
        return chat_ui_msgs, msgs_state, "", None

    except RuntimeError as e:
        # rollback last user message
        if msgs_state and msgs_state[-1].get("role") == "user":
            msgs_state.pop()

        msg = str(e)
        if "mat2 must have the same dtype" in msg or ("Half" in msg and "Byte" in msg):
            chat_ui_msgs = _as_messages(chat_ui)
            chat_ui_msgs.append({"role": "user", "content": ui_user_content})
            chat_ui_msgs.append({
                "role": "assistant",
                "content": "⚠️ Vision dtype mismatch. This almost always happens when the image wasn't passed as a PIL RGB image in the model's expected format ([image, text])."
            })
            return chat_ui_msgs, msgs_state, "", None

        chat_ui_msgs = _as_messages(chat_ui)
        chat_ui_msgs.append({"role": "user", "content": ui_user_content})
        chat_ui_msgs.append({"role": "assistant", "content": f"⚠️ Runtime error: {msg}"})
        return chat_ui_msgs, msgs_state, "", None

    except Exception as e:
        # rollback last user message
        if msgs_state and msgs_state[-1].get("role") == "user":
            msgs_state.pop()

        chat_ui_msgs = _as_messages(chat_ui)
        chat_ui_msgs.append({"role": "user", "content": ui_user_content})
        chat_ui_msgs.append({"role": "assistant", "content": f"⚠️ Internal error: {type(e).__name__}({e})"})
        return chat_ui_msgs, msgs_state, "", None



def do_transcribe(mic_audio):
    """
    mic_audio from Gradio Audio: (sr, np.ndarray) or a filepath depending on type.
    We'll use type="filepath" below for simplicity.
    """
    if mic_audio is None:
        return ""
    return transcribe_audio(mic_audio)


def do_generate_image(prompt: str):
    prompt = (prompt or "").strip()
    if not prompt:
        return None
    img = generate_image(prompt, steps=4)
    if img is None:
        return None
    return img

def toggle_reverse_audio(last_audio_path, is_reversed):
    if not last_audio_path:
        return None, False

    if is_reversed:
        # go back to normal
        return last_audio_path, False
    else:
        # play reversed
        rev_path = wav_reverse(last_audio_path)
        return rev_path, True


# -----------------------------
# Gradio UI
# -----------------------------
with gr.Blocks(title="MiniCPM-o-4.5 Multimodal Chatbot (12GB-friendly)") as demo:
    gr.Markdown(
        """
# MiniCPM-o-4.5 Multimodal Chatbot (12GB-friendly)

- Text chat ✅
- Image understanding ✅
- Voice input (Whisper) ✅
- Voice output (offline TTS) ✅
- Optional image generation (Stable Diffusion Turbo) ⚠️ (separate model)

        """
    )
    
    msgs_state = gr.State([])  # MiniCPM messages
    chat_ui = gr.Chatbot(height=420)

    with gr.Row():
        user_text = gr.Textbox(label="Message", placeholder="Type here...", scale=3, lines=1)
        user_image = gr.Image(label="Optional image", type="pil", scale=2)

   
    with gr.Row():
        reverse_btn = gr.Button("Reverse Audio", variant="huggingface")
        send_btn = gr.Button("Send", variant="primary")
        clear_btn = gr.Button("Clear",variant="stop")
        
        
        
  

    out_audio = gr.Audio(label="Assistant voice", autoplay=True)

  
    with gr.Row():
        with gr.Column(scale=0.2):
            tts_volume = gr.Slider(0.2, 2.0, value=0.5, step=0.05, label="TTS Volume")
        with gr.Column(scale=0.2):
            speak_back = gr.Checkbox(value=True, label="Speak responses (TTS)")
            reverse_after = gr.Checkbox(value=False, label="Play reverse")
        


    with gr.Blocks():
        # Horizontal line separator
         gr.HTML("<hr style='border: 1px solid #bbb; margin: 80px 0;'>")


    last_audio_state = gr.State(None)         # path to the most recent NORMAL audio
    is_reversed_state = gr.State(False)       # whether we're currently playing reversed

    gr.Markdown("## Optional: Image generation (Text → Image)")
    with gr.Row():
        img_prompt = gr.Textbox(label="Image prompt", placeholder="A robot reading a book, cinematic lighting", lines=1)
        gen_img_btn = gr.Button("Generate image", variant="primary")
        clear_btn_img = gr.Button("Clear",variant="stop")
    gen_img_out = gr.Image(label="Generated image")


    with gr.Blocks():
        # Horizontal line separator
         gr.HTML("<hr style='border: 1px solid #bbb; margin: 80px 0;'>")


    gr.Markdown("## Voice input (Speech → Text)")
    with gr.Row():
        mic = gr.Audio(sources=["microphone"], type="filepath", label="Record voice")
        transcribe_btn = gr.Button("Transcribe → Put into message box")

    def _clear():
        chat_ui = []
        msgs_state = []
        user_text = ""
        out_audio = None
        user_image = None
        last_audio_state = None
        is_reversed_state = None
        return chat_ui, msgs_state, user_text, out_audio, user_image, last_audio_state, is_reversed_state
    
    def _clear_img():
        gen_img_out = None
        img_prompt = ""
        return gen_img_out, img_prompt



    send_btn.click(
        fn=chat_step,
        inputs=[chat_ui, msgs_state, user_text, user_image, speak_back, tts_volume, reverse_after, last_audio_state],
        outputs=[chat_ui, msgs_state, user_text, out_audio, last_audio_state, is_reversed_state],
    )

    user_text.submit(
        fn=chat_step,
        inputs=[chat_ui, msgs_state, user_text, user_image, speak_back, tts_volume, reverse_after, last_audio_state],
        outputs=[chat_ui, msgs_state, user_text, out_audio, last_audio_state, is_reversed_state],
    )


    transcribe_btn.click(
        fn=do_transcribe,
        inputs=[mic],
        outputs=[user_text],
    )

    gen_img_btn.click(
        fn=do_generate_image,
        inputs=[img_prompt],
        outputs=[gen_img_out],
    )

    img_prompt.submit(
        fn=do_generate_image,
        inputs=[img_prompt],
        outputs=[gen_img_out],
    )


    clear_btn.click(
        fn=_clear,
        inputs=[],
        outputs=[chat_ui, msgs_state, user_text, out_audio, user_image, last_audio_state, is_reversed_state],
    )

    clear_btn_img.click(
        fn=_clear_img,
        inputs=[],
        outputs=[gen_img_out, img_prompt],
    )


    

    reverse_btn.click(
        fn=toggle_reverse_audio,
        inputs=[last_audio_state, is_reversed_state],
        outputs=[out_audio, is_reversed_state],
    )




if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
