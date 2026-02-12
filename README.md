# MiniCPM-o-4.5 Multimodal Chatbot (12GB-Friendly)

A local, GPU-efficient multimodal chatbot built around MiniCPM-o-4.5, supporting text, image understanding, voice input, offline text-to-speech, and optional image generation — all runnable on a single 12GB GPU.

This project focuses on practical multimodal interaction while keeping memory usage under control through careful model loading, CPU/GPU separation, and optional components.

<br><br>

## Features

- No external APIs required (fully local)

- Text chat with MiniCPM-o-4.5
- Speech-to-text (offline, CPU-based Whisper)
- Text-to-speech (offline Piper TTS)
- Audio reversal & volume control
- Image Generation with SDXL normal, turboXL, or turbo
- Img-Img pipelines
- Image understanding (vision + language)
- 12GB-GPU friendly design

<br><br>

## Architecture Overview
- MiniCPM-o-4.5 (INT4)
  - Runs on GPU via device_map="auto"
- Speech-to-Text
  - faster-whisper on CPU (keeps GPU free)
- Text-to-Speech
  - Piper (offline ONNX voice model)
- Image Generation 
  - Stable Diffusion Normal, TurboXL, or Turbo
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

#### From Windows CMD:
_Note: If this hangs, press CTRL+C and rerun._
```
wsl --install -d Ubuntu-24.04 --name MiniCPM-demo
```

Create a User Name and Password for WSL

<br>

#### Close CMD and Open MiniCPM-demo

<br>

```bash
# 0. Create folder and clone repo
cd ~
git clone https://github.com/amill288/AI-Local-Sandbox.git
cd AI-Local-Sandbox

# 1. Create venv
sudo apt update
sudo apt install -y python3-venv
python3 -m venv .venv
source .venv/bin/activate

# 2. Install PyTorch first (CUDA-specific)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

sudo apt install -y ffmpeg

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

cd ~/AI-Local-Sandbox
```

Feel free to switch the wget to a different `piper` voice model.

<br><br>


## After seleting a script below, Launch the App

Browser Address (use localhost, not 0.0.0.0)
```
http://localhost:7860
```

<br><br>

## Running the App
```
cd ~/AI-Local-Sandbox
python minicpm.py
```

## Stream Webcam Video

```
cd ~/AI-Local-Sandbox
python webcam_live_gradio.py
```


## Generate Images Safely

```
cd ~/AI-Local-Sandbox
python sdxl_safe.py
```


## Generate Images Custom

```
cd ~/AI-Local-Sandbox
python sdxl.py
```


<br><br>

The first run will auto download the required safetensors, after that it will use checkpoints and run more quickly.

<br><br>

## Start Fresh (If need be)
If something happened and you want to start over, from Windows CMD:
<br>

```
wsl --unregister MiniCPM-demo
```
<br>

Then start back at the top.

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
