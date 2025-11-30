#!/usr/bin/env python3
"""
Test script: Compare image editing effects of two models using StreamDiffusion
1. SD-Turbo (StreamDiffusion)
2. SD1.5 + LCM-LoRA (StreamDiffusion)

Supports single image or video input

Note: StreamDiffusion does not perform inversion, it directly preprocesses the image and generates (equivalent to strength=1.0)
"""

import argparse
import sys
from pathlib import Path
import torch
import cv2
import numpy as np
from PIL import Image

# Add project path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.wrapper import StreamDiffusionWrapper
from diffusers.utils import load_image


def extract_frames(video_path, max_frames=None, stride=1):
    """Extract frames from video"""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
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


def numpy_to_pil(img_np, size=None):
    """Convert numpy array to PIL Image"""
    img = Image.fromarray(img_np)
    if size is not None:
        img = img.resize(size, Image.BILINEAR)
    return img


def load_models(device, dtype, resolution, acceleration="xformers", engine_dir="engines"):
    """Load two StreamDiffusion models"""
    models = {}
    
    # ========== Model 1: SD-Turbo (StreamDiffusion) ==========
    print("\n" + "="*60)
    print("Loading Model 1: SD-Turbo (StreamDiffusion)")
    print("="*60)
    try:
        stream_turbo = StreamDiffusionWrapper(
            model_id_or_path="stabilityai/sd-turbo",
            t_index_list=[35],  # 1-step inference
            lora_dict=None,
            mode="img2img",
            output_type="pil",
            lcm_lora_id=None,
            vae_id=None,
            device="cuda" if device.type == "cuda" else "cpu",
            dtype=dtype,
            frame_buffer_size=1,
            width=resolution,
            height=resolution,
            warmup=10,
            acceleration=acceleration,
            do_add_noise=True,  # StreamDiffusion's do_add_noise only adds noise, not inversion
            use_lcm_lora=False,  # SD-Turbo doesn't need LCM-LoRA
            use_tiny_vae=False,
            use_denoising_batch=True,
            cfg_type="none",  # Turbo uses "none"
            seed=2,
            use_safety_checker=False,
            engine_dir=engine_dir,
        )
        
        # Prepare model
        stream_turbo.prepare(
            prompt="",  # Initial prompt, will be updated later
            negative_prompt="",
            num_inference_steps=50,  # StreamDiffusion uses 50 steps internally, actual inference determined by t_index_list
            guidance_scale=0.0,  # SD-Turbo uses 0.0
        )
        
        print("✅ SD-Turbo StreamDiffusion loaded")
        models["turbo"] = stream_turbo
    except Exception as e:
        print(f"❌ Error loading SD-Turbo: {e}")
        import traceback
        traceback.print_exc()
        models["turbo"] = None

    # ========== Model 2: SD1.5 + LCM-LoRA (StreamDiffusion) ==========
    print("\n" + "="*60)
    print("Loading Model 2: SD1.5 + LCM-LoRA (StreamDiffusion)")
    print("="*60)
    try:
        stream_sd15_lcm = StreamDiffusionWrapper(
            model_id_or_path="runwayml/stable-diffusion-v1-5",
            t_index_list=[35],  # 1-step inference
            lora_dict=None,
            mode="img2img",
            output_type="pil",
            lcm_lora_id=None,  # None means use default "latent-consistency/lcm-lora-sdv1-5"
            vae_id=None,
            device="cuda" if device.type == "cuda" else "cpu",
            dtype=dtype,
            frame_buffer_size=1,
            width=resolution,
            height=resolution,
            warmup=10,
            acceleration=acceleration,
            do_add_noise=True,  # StreamDiffusion's do_add_noise only adds noise, not inversion
            use_lcm_lora=True,  # Use LCM-LoRA
            use_tiny_vae=False,
            use_denoising_batch=True,
            cfg_type="none",  # SD1.5+LCM uses "self" (RCFG)
            seed=2,
            use_safety_checker=False,
            engine_dir=engine_dir,
        )
        
        # Prepare model
        stream_sd15_lcm.prepare(
            prompt="",  # Initial prompt, will be updated later
            negative_prompt="",
            num_inference_steps=50,  # StreamDiffusion uses 50 steps internally, actual inference determined by t_index_list
            guidance_scale=1.2,  # SD1.5+LCM uses 1.2
        )
        
        print("✅ SD1.5 + LCM-LoRA StreamDiffusion loaded")
        models["sd15_lcm"] = stream_sd15_lcm
    except Exception as e:
        print(f"❌ Error loading SD1.5+LCM: {e}")
        import traceback
        traceback.print_exc()
        models["sd15_lcm"] = None
    
    return models


def process_single_image(image_path, models, prompt, resolution, output_dir):
    """Process single image (using StreamDiffusion)"""
    print(f"\nLoading image from: {image_path}")
    init_image = load_image(image_path).resize((resolution, resolution))
    print(f"✅ Image loaded: mode={init_image.mode}, size={init_image.size}")
    
    # Save original image
    orig_path = output_dir / "original.png"
    init_image.save(orig_path)
    print(f"✅ Original image saved to: {orig_path}")
    
    results = {}
    
    # Model 1: SD-Turbo
    if models["turbo"] is not None:
        print(f"\nRunning SD-Turbo inference with StreamDiffusion...")
        print("   Note: StreamDiffusion does NOT perform inversion (no strength parameter)")
        try:
            # StreamDiffusion call pattern
            image_tensor = models["turbo"].preprocess_image(init_image)
            image_turbo = models["turbo"](image=image_tensor, prompt=prompt)
            
            # Ensure return is PIL Image
            if isinstance(image_turbo, list):
                image_turbo = image_turbo[0]
            
            output_path = output_dir / "sd_turbo.png"
            image_turbo.save(output_path)
            results["SD-Turbo"] = output_path
            print(f"✅ Saved to: {output_path}")
        except Exception as e:
            print(f"❌ Error with SD-Turbo inference: {e}")
            import traceback
            traceback.print_exc()
    
    # Model 2: SD1.5 + LCM-LoRA
    if models["sd15_lcm"] is not None:
        print(f"\nRunning SD1.5+LCM inference with StreamDiffusion...")
        print("   Note: StreamDiffusion does NOT perform inversion (no strength parameter)")
        try:
            # StreamDiffusion call pattern
            image_tensor = models["sd15_lcm"].preprocess_image(init_image)
            image_sd15_lcm = models["sd15_lcm"](image=image_tensor, prompt=prompt)
            
            # Ensure return is PIL Image
            if isinstance(image_sd15_lcm, list):
                image_sd15_lcm = image_sd15_lcm[0]
            
            output_path = output_dir / "sd15_lcm.png"
            image_sd15_lcm.save(output_path)
            results["SD1.5+LCM-LoRA"] = output_path
            print(f"✅ Saved to: {output_path}")
        except Exception as e:
            print(f"❌ Error with SD1.5+LCM inference: {e}")
            import traceback
            traceback.print_exc()
    
    return results


def process_video(video_path, models, prompt, resolution, output_dir, max_frames=None, stride=1):
    """Process video (using StreamDiffusion)"""
    print(f"\nExtracting frames from video: {video_path}")
    frames = extract_frames(video_path, max_frames=max_frames, stride=stride)
    print(f"✅ Extracted {len(frames)} frames")
    
    # Create output directory structure
    input_dir = output_dir / "input_frames"
    output_turbo_dir = output_dir / "output_turbo"
    output_sd15_lcm_dir = output_dir / "output_sd15_lcm"
    
    input_dir.mkdir(parents=True, exist_ok=True)
    if models["turbo"] is not None:
        output_turbo_dir.mkdir(parents=True, exist_ok=True)
    if models["sd15_lcm"] is not None:
        output_sd15_lcm_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each frame
    for frame_idx, frame_np in enumerate(frames):
        print(f"\nProcessing frame {frame_idx + 1}/{len(frames)}")
        
        # Convert to PIL Image and resize
        frame_pil = numpy_to_pil(frame_np, size=(resolution, resolution))
        
        # Save input frame
        input_path = input_dir / f"frame_{frame_idx:05d}.png"
        frame_pil.save(input_path)
        
        # Model 1: SD-Turbo
        if models["turbo"] is not None:
            try:
                image_tensor = models["turbo"].preprocess_image(frame_pil)
                output_turbo = models["turbo"](image=image_tensor, prompt=prompt)
                if isinstance(output_turbo, list):
                    output_turbo = output_turbo[0]
                
                output_path = output_turbo_dir / f"frame_{frame_idx:05d}.png"
                output_turbo.save(output_path)
                print(f"  ✅ SD-Turbo: {output_path}")
            except Exception as e:
                print(f"  ❌ Error with SD-Turbo: {e}")
        
        # Model 2: SD1.5 + LCM-LoRA
        if models["sd15_lcm"] is not None:
            try:
                image_tensor = models["sd15_lcm"].preprocess_image(frame_pil)
                output_sd15_lcm = models["sd15_lcm"](image=image_tensor, prompt=prompt)
                if isinstance(output_sd15_lcm, list):
                    output_sd15_lcm = output_sd15_lcm[0]
                
                output_path = output_sd15_lcm_dir / f"frame_{frame_idx:05d}.png"
                output_sd15_lcm.save(output_path)
                print(f"  ✅ SD1.5+LCM: {output_path}")
            except Exception as e:
                print(f"  ❌ Error with SD1.5+LCM: {e}")
    
    print(f"\n✅ Video processing completed!")
    print(f"Input frames saved to: {input_dir}")
    if models["turbo"] is not None:
        print(f"SD-Turbo outputs saved to: {output_turbo_dir}")
    if models["sd15_lcm"] is not None:
        print(f"SD1.5+LCM outputs saved to: {output_sd15_lcm_dir}")


def main():
    parser = argparse.ArgumentParser(description="Compare two models using StreamDiffusion")
    parser.add_argument("--image", type=str, default=None, help="Input image path or URL")
    parser.add_argument("--video", type=str, default=None, help="Input video path")
    parser.add_argument("--prompt", type=str, required=True, help="Edit instruction")
    parser.add_argument("--output-dir", type=str, default="output_comparison", help="Output directory")
    parser.add_argument("--resolution", type=int, default=512, help="Image resolution (512 or 768)")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda or cpu)")
    parser.add_argument("--fp16", action="store_true", help="Use FP16 precision")
    parser.add_argument("--acceleration", type=str, default="xformers", choices=["none", "xformers", "tensorrt"], help="Acceleration method")
    parser.add_argument("--engine-dir", type=str, default="engines", help="Engine directory for TensorRT")
    parser.add_argument("--max-frames", type=int, default=None, help="Maximum number of frames to process from video")
    parser.add_argument("--stride", type=int, default=1, help="Frame stride for video processing")
    
    args = parser.parse_args()
    
    # Validate input
    if args.image is None and args.video is None:
        parser.error("Either --image or --video must be provided")
    if args.image is not None and args.video is not None:
        parser.error("Please provide either --image or --video, not both")

    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if args.fp16 and device.type == "cuda" else torch.float32
    
    print("="*60)
    if args.image:
        print("Image Editing Comparison: 2 Models (StreamDiffusion)")
    else:
        print("Video Editing Comparison: 2 Models (StreamDiffusion)")
    print("="*60)
    print(f"Device: {device}, dtype: {dtype}")
    print(f"Resolution: {args.resolution}x{args.resolution}")
    print(f"Acceleration: {args.acceleration}")
    print(f"Edit prompt: {args.prompt}")
    print(f"⚠️  Note: StreamDiffusion does NOT perform inversion (no strength parameter)")
    print(f"   It directly preprocesses the image and generates (equivalent to strength=1.0)")
    if args.video:
        print(f"Max frames: {args.max_frames if args.max_frames else 'all'}")
        print(f"Frame stride: {args.stride}")
    print("="*60)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load models
    models = load_models(device, dtype, args.resolution, args.acceleration, args.engine_dir)
    
    # Process input
    if args.image:
        results = process_single_image(
            args.image, models, args.prompt, 
            args.resolution, output_dir
        )
        
        # Summary
        print("\n" + "="*60)
        print("Comparison completed!")
        print("="*60)
        print(f"Original image: {output_dir / 'original.png'}")
        print("\nResults:")
        for model_name, output_path in results.items():
            print(f"  {model_name}: {output_path}")
        print("="*60)
    else:
        process_video(
            args.video, models, args.prompt,
            args.resolution, output_dir, 
            max_frames=args.max_frames, stride=args.stride
        )


if __name__ == "__main__":
    main()
