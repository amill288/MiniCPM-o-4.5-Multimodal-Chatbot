import os
import time
import threading
from dataclasses import dataclass
from typing import Optional

import gradio as gr
import numpy as np
from PIL import Image

import torch
from transformers import AutoModel, AutoTokenizer

from dataclasses import dataclass

# ----------------------------
# Config
# ----------------------------
MODEL_ID = os.environ.get("MODEL_ID", "openbmb/MiniCPM-V-4_5-int4")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Reasonable default for live-ish analysis (not camera FPS)
DEFAULT_ANALYZE_FPS = 2.0  # analyze every 0.5s
DEFAULT_DROP_IF_BUSY = True
DEFAULT_TOKEN_LIMIT = 120

# Resize frames before inference for speed/VRAM stability.
# 640 long-edge is a good starting point.
MAX_LONG_EDGE = 640


# ----------------------------
# Model load (once)
# ----------------------------
print(f"[live] Loading model: {MODEL_ID} on {DEVICE} ...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

# Keep dtype consistent with your existing approach:
# - float16 on CUDA
# - float32 on CPU (more stable)
dtype = torch.float16 if DEVICE == "cuda" else torch.float32

model = AutoModel.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    torch_dtype=dtype
).to(DEVICE)

model.eval()
print("[live] Model loaded.")


# ----------------------------
# Helpers
# ----------------------------
def _to_pil(frame) -> Optional[Image.Image]:
    """Convert Gradio webcam frame (numpy or PIL) to PIL RGB."""
    if frame is None:
        return None

    if isinstance(frame, Image.Image):
        frame = np.fliplr(frame)
        img = frame
    else:
        # Gradio webcam frames are usually numpy arrays (H, W, 3) uint8
        if isinstance(frame, np.ndarray):
            frame = np.fliplr(frame)
            if frame.ndim == 2:
                img = Image.fromarray(frame).convert("RGB")
            elif frame.ndim == 3:
                img = Image.fromarray(frame).convert("RGB")
            else:
                return None
        else:
            return None

    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def _resize_long_edge(img: Image.Image, max_long_edge: int = MAX_LONG_EDGE) -> Image.Image:
    w, h = img.size
    long_edge = max(w, h)
    if long_edge <= max_long_edge:
        return img

    scale = max_long_edge / float(long_edge)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return img.resize((new_w, new_h), Image.BICUBIC)


@dataclass
class LiveState:
    running: bool = False
    busy: bool = False
    last_text: str = ""
    last_ok_ts: float = 0.0
    frames_seen: int = 0
    frames_analyzed: int = 0


_state_lock = threading.Lock()
_stop_event = threading.Event()
_latest_frame = None
_latest_frame_lock = threading.Lock()


def end_session(state: LiveState):
    state.running = False
    _stop_event.set()
    return "⛔️ Stopped", state



def capture_latest_frame(frame, state: LiveState):
    global _latest_frame
    state.frames_seen += 1
    with _latest_frame_lock:
        _latest_frame = frame
    return f"🟢 Captured. Seen: {state.frames_seen} | Analyzed: {state.frames_analyzed}", state



def run_session(instruction, analyze_fps, drop_if_busy, state: LiveState, token_limit):
    global _latest_frame

    state.running = True
    _stop_event.clear()

    while state.running and not _stop_event.is_set():
        # Grab latest frame
        with _latest_frame_lock:
            frame = _latest_frame

        if frame is None:
            yield state.last_text, "Waiting for webcam…", state
            time.sleep(0.05)
            continue

        # Don’t overlap inference
        with _state_lock:
            if state.busy:
                if drop_if_busy:
                    time.sleep(0.01)
                    continue
                time.sleep(0.01)
                continue
            state.busy = True

        try:
            pil = _to_pil(frame)
            if pil is None:
                yield state.last_text, "Bad frame; waiting…", state
                time.sleep(0.05)
                continue

            pil_small = _resize_long_edge(pil, MAX_LONG_EDGE)
            prompt = (instruction or "").strip() or "Describe what you see."
            msgs = [{"role": "user", "content": prompt}]

            # Streaming path (MiniCPM requires sampling=True)
            if hasattr(model, "stream_chat"):
                partial = ""
                with torch.no_grad():
                    for chunk in model.stream_chat(
                        image=pil_small,
                        msgs=msgs,
                        tokenizer=tokenizer,
                        sampling=True,
                        temperature=0.1,
                        top_p=1.0,
                        max_new_tokens=token_limit,
                        max_slice_nums=1,
                    ):
                        # parse chunk robustly
                        if isinstance(chunk, tuple) and len(chunk) >= 1:
                            text = chunk[0]
                        elif isinstance(chunk, dict) and "text" in chunk:
                            text = chunk["text"]
                        else:
                            text = chunk
                        text = text if isinstance(text, str) else str(text)

                        # usually "full text so far"
                        partial = text if len(text) >= len(partial) else partial + text
                        state.last_text = partial
                        yield state.last_text, "👁‍🗨 Streaming…", state

            else:
                # Fallback no-token-stream
                with torch.no_grad():
                    ans = model.chat(
                        image=pil_small,
                        msgs=msgs,
                        tokenizer=tokenizer,
                        sampling=False,
                        max_new_tokens=token_limit,
                        max_slice_nums=1,
                    )
                state.last_text = ans
                yield state.last_text, "🟢 Running...", state

            state.frames_analyzed += 1
            state.last_ok_ts = time.time()
            yield state.last_text, "✅ OK", state

        except (GeneratorExit, BrokenPipeError, ConnectionResetError):
            return
        except Exception as e:
            yield state.last_text, f"Error: {type(e).__name__}: {e}", state
        finally:
            with _state_lock:
                state.busy = False

        fps = float(analyze_fps) if analyze_fps else 2.0
        time.sleep(max(0.05, 1.0 / max(0.1, fps)))

    yield state.last_text, "⛔️ Session stopped.", state






# ----------------------------
# Gradio UI
# ----------------------------
with gr.Blocks(title="Live Webcam (Near-Live) MiniCPM",
    ) as demo:
    gr.Markdown("## Live Webcam → Near-Live Vision Chat (throttled, latest-frame-wins)")

    state = gr.State(LiveState())
    with gr.Column():
        with gr.Row():
            btn_start = gr.Button("Start Session", variant="primary")
            btn_end = gr.Button("End Session", variant="stop")

        with gr.Row():
            webcam = gr.Image(
                sources=["webcam"],
                type="numpy",
                streaming=True,
                label="Webcam (streaming)",
                elem_id="webcam-mirror", 
                
            )


            with gr.Column():
                instruction = gr.Textbox(
                    label="Instruction (prompt used each analysis tick)",
                    value="Keep responses under 100 characters. Describe what you see, and mention any changes from the previous moment.",
                    lines=3
                )

                analyze_fps = gr.Slider(
                    minimum=0.5,
                    maximum=5.0,
                    value=DEFAULT_ANALYZE_FPS,
                    step=0.5,
                    label="Analyze FPS (how often to run the model)"
                )

                token_limit = gr.Slider(
                    minimum=10,
                    maximum=500,
                    value=DEFAULT_TOKEN_LIMIT,
                    step=5,
                    label="Token Limit (Length of Response)"
                )

                drop_if_busy = gr.Checkbox(
                    value=DEFAULT_DROP_IF_BUSY,
                    label="Drop frames if model is busy (recommended)"
                )

                status = gr.Textbox(label="Status", interactive=False)
                output = gr.Textbox(label="Model output", lines=10, interactive=False)

    # Stream handler: every incoming frame calls this function; it throttles internally.
    
    webcam.stream(
        capture_latest_frame,
        inputs=[webcam, state],
        outputs=[status, state],
        show_progress=False,
        )
    
    
    JS_START_WEBCAM = r"""
    (...args) => {
    const tryClick = () => {
        const root = document.querySelector('#webcam-mirror');
        const btn = root?.querySelector('.button-wrap button');
        if (btn) { btn.click(); return true; }
        return false;
    };

    if (tryClick()) return args;

    let n = 0;
    const t = setInterval(() => {
        if (tryClick() || ++n > 20) clearInterval(t);
    }, 50);

    return args;
    }
    """



    JS_STOP_WEBCAM = r"""
    (...args) => {
    const root = document.querySelector('#webcam-mirror');
    if (root) {
        const btn = root.querySelector('.button-wrap button');
        if (btn) btn.click();
    }
    return args;  // <-- CRITICAL
    }
    """


    btn_start.click(
        fn = run_session,
        inputs=[instruction, analyze_fps, drop_if_busy, state, token_limit],
        outputs=[output, status, state],
        js=JS_START_WEBCAM,  
        show_progress=False,
        
    )

    btn_end.click( 
        fn = end_session,
        inputs=[state],
        outputs=[status, state],
        js=JS_STOP_WEBCAM, 
        show_progress=False
    )





JS = r"""
() => {
  if ('caches' in window) {
    caches.keys().then(names => {
      for (const name of names) {
        caches.delete(name);
      }
    });
  }
}
"""
demo.launch(js=JS)



CSS = """
video.svelte-1tktvmr.flip {
    transform: none !important;
}

"""

# Queue settings:
# - default_concurrency_limit=1 helps ensure you don't get overlapping calls
# - max_size=1 reduces backlog
demo.queue(default_concurrency_limit=1, max_size=1).launch(css=CSS, js=JS)
