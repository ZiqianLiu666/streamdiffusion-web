# StreamDiffusion-Web

Course project for **CSC_51073_EP – Analyse d'Image et Vision par Ordinateur**  
Realtime "Speak & Draw" system based on StreamDiffusion, Stable Diffusion Turbo and real-time matting.

![demo](./visualization/Video-min.gif)

## Overview

**Realtime SpeakDraw** is a real-time image-to-image system that lets you:

- Stream video frames from the browser / camera to a backend.
- Control a **Stable Diffusion Turbo** (SDXL-Turbo) img2img pipeline with text prompts and image prompts (IP-Adapter).
- Optionally apply **RVM (Robust Video Matting)** to only modify the human region (“human-only edit”).
- Use a speech-to-text endpoint (Whisper-like) to speak prompts instead of typing them (You can choose to type the prompt if you don't mind the tiredness :).

The backend is implemented with **FastAPI + WebSocket** and uses **StreamDiffusion** to achieve high-FPS diffusion with SDXL-Turbo, TinyVAE and xformers acceleration.

## Models & Dependencies

- **Stable Diffusion Turbo** – `stabilityai/sd-turbo`
- **StreamDiffusion** – `real-time diffusion acceleration`
- **RVM (Robust Video Matting)** – `human matting`
- **Whisper (open-source)** – `speech-to-text`
- **TinyVAE (TAESD)** – `lightweight VAE for fast decoding`

## Main Features

- **Real-time img2img with StreamDiffusion**
  - Based on SD-Turbo (`stabilityai/sd-turbo`) by default.
  - Optimized for streaming: denoising batch, frame buffer, partial steps, etc.
- **Two editing modes**
  - `full` – run img2img on the whole frame.
  - `human` – use RVM to extract a soft alpha matte and only replace the human foreground, keeping the original background.
- **Speech-to-text integration**
  - `Speech2txt` -  Speech-to-Text based on open sourced Whisper.
- **FastAPI REST + WebSocket API**
  - WebSocket channel for control and frame upload.
  - MJPEG streaming endpoint for generated frames.
  - JSON settings endpoint for front-end auto-configuration.
- **Acceleration & optimizations**
  - Optional **TinyVAE** (`madebyollin/taesd`) for faster VAE.
  - Support for **xFormers** and **TensorRT** acceleration engines.

## Environment Setup

This project uses a Conda environment defined in **environment.yml**.
```bash
conda env create -f environment.yml
conda activate cv_project
```

## Launch command
```bash
python main.py
```

## 📌 Acknowledgements

This project is mainly built upon the following open-source repositories:

- **[StreamDiffusion](https://github.com/cumulo-autumn/StreamDiffusion)**
- **[OpenAI Whisper](https://github.com/openai/whisper)**
- **[Robust Video Matting (RVM)](https://github.com/PeterL1n/RobustVideoMatting)**
- **[IP-Adapter](https://github.com/tencent-ailab/IP-Adapter)**

We integrate these components to build a real-time img2img pipeline with audio input, video matting, and image-conditioned control.

## 🚧 Future Work

We plan to incorporate the training strategy proposed in the SwiftEdit project to address the limitation that StreamDiffusion cannot perform fine-grained local edits.

- **[SwiftEdit](https://github.com/lucidrains/swiftedit)**


