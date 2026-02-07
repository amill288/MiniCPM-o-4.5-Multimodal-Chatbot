# MiniCPM-V Multimodal Chatbot (12GB-Friendly)

A local, GPU-efficient multimodal chatbot built around MiniCPM-V-4.5, supporting text, image understanding, voice input, offline text-to-speech, and optional image generation — all runnable on a single 12GB GPU.

This project focuses on practical multimodal interaction while keeping memory usage under control through careful model loading, CPU/GPU separation, and optional components.

<br><br>

## Features

💬 Text chat with MiniCPM-V

🖼️ Image understanding (vision + language)

🎤 Speech-to-text (offline, CPU-based Whisper)

🔊 Text-to-speech (offline Piper TTS)

🔁 Audio reversal & volume control

🎨 Optional text-to-image generation (Stable Diffusion Turbo)

⚡ 12GB-GPU friendly design

🧠 No external APIs required (fully local)

<br><br>

## Architecture Overview
- MiniCPM-V-4.5 (INT4)
  - Runs on GPU via device_map="auto"
- Speech-to-Text
  - faster-whisper on CPU (keeps GPU free)
- Text-to-Speech
  - Piper (offline ONNX voice model)
- Image Generation (Optional)
  - Stable Diffusion Turbo
  - Loaded on CPU and moved to GPU only during inference
- UI
  - Gradio Blocks interface

This separation allows smooth interaction even on consumer GPUs.

<br><br>

## Requirements

### Hardware
- NVIDIA GPU with ~12GB VRAM (tested)
- CPU capable of running Whisper + Piper

Software
- Python 3.10+
- CUDA-enabled PyTorch
- Linux recommended (WSL2 works well)

<br><br>
## Installation

```bash
git clone https://github.com/amill288/MiniCPM-o-4.5_Multimodal-Chatbot.git
cd MiniCPM

# 1. Create venv
python -m venv .venv
source .venv/bin/activate

# 2. Install PyTorch first (CUDA-specific)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 3. Install the rest
pip install -r requirements.txt

```
<br>

You’ll also need:
- A Piper voice model (ONNX)
- `piper` available on your PATH

<br>

```
mkdir -p ~/piper_voices/libritts_r_medium
cd ~/piper_voices/libritts_r_medium

wget -O en_US-libritts_r-medium.onnx \
  https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/libritts_r/medium/en_US-libritts_r-medium.onnx

wget -O en_US-libritts_r-medium.onnx.json \
  https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/libritts_r/medium/en_US-libritts_r-medium.onnx.json
```

Feel free to switch the wget to a different `piper` voice model.

<br><br>


## Running the App
```
python minicpm.py
```
The Gradio interface will launch on:
```
http://localhost:7860
```
<br><br>

## Usage Notes
- Voice input/output is fully offline
- Image generation is optional and can be disabled if VRAM is tight
- Audio playback supports:
  - Volume adjustment
  - Instant reversal toggle (normal ↔ reversed)

<br><br>

## License

This project is licensed under the MIT License.

Important:
This repository provides application code and UI logic only.
Model weights and third-party tools (MiniCPM, Stable Diffusion, Whisper, Piper, etc.) are governed by their own respective licenses.

See the LICENSE file for details.

<br><br>

## Acknowledgements
- OpenBMB — MiniCPM-V
- Hugging Face — Transformers & Diffusers
- OpenAI — Whisper (via faster-whisper)
- Gradio — UI framework
- Piper TTS — Offline speech synthesis

<br><br>

## Disclaimer
This project is intended for research, experimentation, and educational use.
Always review the licenses of included models before using in production.


Note: This repository provides a wrapper and UI for third-party models and tools.
Model weights and external dependencies are governed by their own respective licenses.
