import argparse
import time
import statistics
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import sys
import torch
from transformers import CLIPProcessor, CLIPModel

# Add repository root to sys.path for importing utils.wrapper
root = Path(__file__).resolve().parents[1]
sys.path.append(str(root))

from utils.wrapper import StreamDiffusionWrapper  # noqa: E402


# ============ Video frame reading ============

def extract_frames(video_path, max_frames=None, stride=1):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    frames = []
    idx = 0
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        if idx % stride == 0:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
            if max_frames is not None and len(frames) >= max_frames:
                break
        idx += 1
    cap.release()
    return frames


def numpy_to_pil(img_np, size=(512, 512)):
    img = Image.fromarray(img_np)
    if size is not None:
        img = img.resize(size, Image.BILINEAR)
    return img


# ============ CLIP：text-image CLIPScore ============

class CLIPScorer:
    """
    Does one thing:
    - Encode prompt into text_emb
    - Encode edited frame into image_emb
    - CLIPScore = cos(image_emb, text_emb)
    """

    def __init__(self, device, model_name: str = "openai/clip-vit-large-patch14"):
        self.device = device
        self.model = CLIPModel.from_pretrained(model_name).to(device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.text_emb = None  # [1, D]

    def set_prompt(self, prompt: str):
        with torch.no_grad():
            inputs = self.processor(
                text=[prompt],
                images=None,
                return_tensors="pt",
                padding=True,
            ).to(self.device)
            text_features = self.model.get_text_features(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            )
            self.text_emb = text_features / text_features.norm(dim=-1, keepdim=True)

    def score_image(self, image_pil: Image.Image) -> float:
        assert self.text_emb is not None, "Call set_prompt() first."

        with torch.no_grad():
            inputs = self.processor(
                text=None,
                images=image_pil,
                return_tensors="pt",
                padding=True,
            ).to(self.device)
            image_features = self.model.get_image_features(
                pixel_values=inputs["pixel_values"]
            )
            image_emb = image_features / image_features.norm(dim=-1, keepdim=True)

        sim = float((image_emb @ self.text_emb.T).squeeze().item())
        return sim


# ============ Build Turbo's StreamDiffusionWrapper ============

def build_stream_wrapper_turbo(
    device: torch.device,
    dtype: torch.dtype,
    acceleration: str = "xformers",
    use_tiny_vae: bool = False,
    enable_similar_image_filter: bool = False,
    similar_image_filter_threshold: float = 0.98,
    similar_image_filter_max_skip_frame: int = 10,
    width: int = 512,
    height: int = 512,
    engine_dir: str = "engines",
    cfg_type: str = "none"
):
    """
    Wrapper specifically for stabilityai/sd-turbo.
    Controlled by parameters:
    - acceleration: "none" / "xformers" / "tensorrt"
    - use_tiny_vae: TinyVAE (taesd)
    - enable_similar_image_filter: skip similar frames
    """
    device_str = "cuda" if device.type == "cuda" else "cpu"

    wrapper = StreamDiffusionWrapper(
        model_id_or_path="stabilityai/sd-turbo",
        t_index_list=[35, 45],          # Two t indices, equivalent to 2 denoising steps
        lora_dict=None,
        mode="img2img",
        output_type="pil",
        lcm_lora_id=None,
        vae_id=None,
        device=device_str,
        dtype=dtype,
        frame_buffer_size=1,
        width=width,
        height=height,
        warmup=10,
        acceleration=acceleration,
        do_add_noise=True,
        device_ids=None,
        use_lcm_lora=False,
        use_tiny_vae=use_tiny_vae,
        enable_similar_image_filter=enable_similar_image_filter,
        similar_image_filter_threshold=similar_image_filter_threshold,
        similar_image_filter_max_skip_frame=similar_image_filter_max_skip_frame,
        use_denoising_batch=True,
        cfg_type=cfg_type,                # Turbo ablation: don't use CFG/RCFG here
        seed=2,
        use_safety_checker=False,
        engine_dir=engine_dir,
    )

    return wrapper


# ============ Benchmark using StreamDiffusionWrapper ============

def benchmark_turbo_with_stream(
    stream: StreamDiffusionWrapper,
    frames,
    prompt: str,
    num_inference_steps: int,
    guidance_scale: float,
    device: torch.device,
    clip_scorer: CLIPScorer,
    warmup: int = 2,
):
    latencies = []
    clip_scores = []

    # Prepare CLIP text embedding
    clip_scorer.set_prompt(prompt)

    # Prepare once (same as img2img.py)
    stream.prepare(
        prompt=prompt,
        negative_prompt="",
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
    )

    # warmup
    if warmup > 0:
        for i in range(min(warmup, len(frames))):
            _ = stream.img2img(numpy_to_pil(frames[i]), prompt=None)

    if torch.cuda.is_available() and device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    # Official timing & CLIPScore
    for frame in frames:
        init_pil = numpy_to_pil(frame)

        if torch.cuda.is_available() and device.type == "cuda":
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()

        edited_pil = stream.img2img(init_pil, prompt=None)

        if torch.cuda.is_available() and device.type == "cuda":
            torch.cuda.synchronize(device)
        t1 = time.perf_counter()
        latencies.append(t1 - t0)

        score = clip_scorer.score_image(edited_pil)
        clip_scores.append(score)

    total_time = sum(latencies)
    n = len(latencies)
    fps = n / total_time if total_time > 0 else 0.0
    avg_lat = statistics.mean(latencies)
    p95_lat = (
        statistics.quantiles(latencies, n=20)[-1] if len(latencies) >= 20 else max(latencies)
    )

    if torch.cuda.is_available() and device.type == "cuda":
        max_mem = torch.cuda.max_memory_allocated(device) / (1024**2)
    else:
        max_mem = 0.0

    results = {
        "num_frames": n,
        "fps": fps,
        "avg_latency_ms": avg_lat * 1000,
        "p95_latency_ms": p95_lat * 1000,
        "max_mem_MB": max_mem,
        "clipscore_mean": statistics.mean(clip_scores),
    }
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--prompt", type=str, required=True)

    parser.add_argument("--max-frames", type=int, default=64)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument(
        "--cfg-type",
        type=str,
        default="none",
        choices=["none", "full", "self", "initialize"],
        help="cfg_type for StreamDiffusionWrapper (Residual CFG mode).",
    )

    # StreamDiffusion.prepare parameters (note: t_index_list is fixed to 2)
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=1.2)

    # Acceleration & component switches
    parser.add_argument(
        "--acceleration",
        type=str,
        default="xformers",
        choices=["none", "xformers", "tensorrt"],
        help="Acceleration backend for StreamDiffusionWrapper",
    )
    parser.add_argument(
        "--taesd",
        action="store_true",
        help="Use TinyVAE (taesd) inside StreamDiffusionWrapper",
    )
    parser.add_argument(
        "--engine-dir",
        type=str,
        default="engines",
        help="TensorRT engine dir (used when acceleration=tensorrt)",
    )
    parser.add_argument(
        "--enable-similar-filter",
        action="store_true",
        help="Enable similar_image_filter inside StreamDiffusionWrapper",
    )
    parser.add_argument(
        "--similar-threshold",
        type=float,
        default=0.98,
        help="similar_image_filter_threshold",
    )
    parser.add_argument(
        "--similar-max-skip",
        type=int,
        default=10,
        help="similar_image_filter_max_skip_frame",
    )

    args = parser.parse_args()

    video_path = Path(args.video)
    frames = extract_frames(video_path, max_frames=args.max_frames, stride=args.stride)
    print(f"Loaded {len(frames)} frames from {video_path}")
    if not frames:
        print("No frames extracted, exit.")
        return

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if args.fp16 and device.type == "cuda" else torch.float32
    torch.backends.cudnn.benchmark = True

    # CLIP scorer
    print("Loading CLIP model for CLIPScore...")
    clip_scorer = CLIPScorer(device=device)

    # Build Turbo's StreamDiffusionWrapper (with component switches)
    stream = build_stream_wrapper_turbo(
        device=device,
        dtype=dtype,
        acceleration=args.acceleration,
        use_tiny_vae=args.taesd,
        enable_similar_image_filter=args.enable_similar_filter,
        similar_image_filter_threshold=args.similar_threshold,
        similar_image_filter_max_skip_frame=args.similar_max_skip,
        width=512,
        height=512,
        engine_dir=args.engine_dir,
        cfg_type=args.cfg_type,
    )

    print(
        f"Benchmarking SD-Turbo (accel={args.acceleration}, "
        f"taesd={args.taesd}, similar_filter={args.enable_similar_filter})..."
    )
    res = benchmark_turbo_with_stream(
        stream=stream,
        frames=frames,
        prompt=args.prompt,
        num_inference_steps=args.num_steps,
        guidance_scale=args.guidance_scale,
        device=device,
        clip_scorer=clip_scorer,
    )

    print("===== Turbo Ablation Results =====")
    for k, v in res.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
