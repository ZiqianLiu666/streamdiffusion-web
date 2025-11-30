import sys
import os

import time

sys.path.append(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
    )
)

from utils.wrapper import StreamDiffusionWrapper

import torch

from config import Args
from pydantic import BaseModel, Field
from PIL import Image
import math

from utils.rvm_masker import RVMMasker
import numpy as np

# base_model = "stabilityai/sd-turbo"
# taesd_model = "madebyollin/taesd"

base_model = "stabilityai/sdxl-turbo"
taesd_model = "madebyollin/taesdxl"

default_prompt = "Portrait of The Joker halloween costume, face painting , glare pose, detailed, intricate, full of colour, cinematic lighting, trending on artstation, 8k, hyperrealistic, focused, extreme details, unreal engine 5 cinematic, masterpiece"
default_negative_prompt = "black and white, blurry, low resolution, pixelated,  pixel art, low quality, low fidelity"

page_content = """
<h1 class="text-3xl font-bold">CSC_51073_EP - Analyse d'Image et Vision par Ordinateur</h1>
<h3 class="text-xl font-bold">Realtime SpeakDraw</h3>
"""


class Pipeline:
    class Info(BaseModel):
        name: str = "StreamDiffusion img2img"
        input_mode: str = "image"
        page_content: str = page_content

    class InputParams(BaseModel):
        prompt: str = Field(
            default_prompt,
            title="Prompt",
            field="textarea",
            id="prompt",
        )
        # negative_prompt: str = Field(
        #     default_negative_prompt,
        #     title="Negative Prompt",
        #     field="textarea",
        #     id="negative_prompt",
        # )
        width: int = Field(
            512, min=2, max=15, title="Width", disabled=True, hide=True, id="width"
        )
        height: int = Field(
            512, min=2, max=15, title="Height", disabled=True, hide=True, id="height"
        )
        ip_adapter_scale: float = Field(
            1.0, min=0.0, max=10.0, step=0.1, title="IP-Adapter Scale", field="range", id="ip_adapter_scale"
        )

    def __init__(self, args: Args, device: torch.device, torch_dtype: torch.dtype):
        params = self.InputParams()
        
        # Automatically select taesd_model based on model type
        is_sdxl = "sdxl" in args.base_model.lower()
        taesd_model = "madebyollin/taesdxl" if is_sdxl else "madebyollin/taesd"
        
        print(f"[Pipeline] Using base_model: {args.base_model}")
        print(f"[Pipeline] Using taesd_model: {taesd_model if args.taesd else 'None (TAESD disabled)'}")
        print(f"[Pipeline] Model size: SDXL" if is_sdxl else "[Pipeline] Model size: SD1.5/SD2.1")
        
        self.stream = StreamDiffusionWrapper(
            model_id_or_path=args.base_model,  # Use base_model from args
            use_tiny_vae=args.taesd,
            device=device,
            dtype=torch_dtype,
            t_index_list=[35],
            frame_buffer_size=1,
            width=params.width,
            height=params.height,
            use_lcm_lora=False,
            output_type="pil",
            warmup=10,
            vae_id=taesd_model if args.taesd else None,
            acceleration=args.acceleration,
            mode="img2img",
            use_denoising_batch=True,
            cfg_type="none",
            use_safety_checker=args.safety_checker,
            # enable_similar_image_filter=True,
            # similar_image_filter_threshold=0.98,
            engine_dir=args.engine_dir,
            
            # SD-Turbo IP-Adapter configuration
            use_ip_adapter=True,
            # Official SDXL IP-Adapter weights (must be .bin/.pt loadable with torch.load; safetensors need conversion)
            ip_adapter_repo_id="h94/IP-Adapter",
            ip_adapter_weight_name="sdxl_models/ip-adapter_sdxl.bin",
            ip_image_encoder_repo_id="h94/IP-Adapter",
            # Can be None initially, updated after frontend uploads reference image
            ip_ref_image=None,
            ip_adapter_scale=1.0,
        )

        self.last_prompt = default_prompt
        self.stream.prepare(
            prompt=default_prompt,
            negative_prompt=default_negative_prompt,
            num_inference_steps=50,
            guidance_scale=1.2,
        )
        
        # RVM
        device_str = "cuda" if device.type == "cuda" else "cpu"
        self.masker = RVMMasker(device=device_str, model_name="resnet50", downsample_ratio=0.25)
    
    def update_ip_ref_image(self, image: Image.Image):
        """Update IP-Adapter reference image"""
        if self.stream.use_ip_adapter:
            self.stream.set_ip_adapter_image(image)


    # def predict(self, params: "Pipeline.InputParams") -> Image.Image:
    #     image_tensor = self.stream.preprocess_image(params.image)
    #     output_image = self.stream(image=image_tensor, prompt=params.prompt)

    #     return output_image
    
    def predict(self, params: "Pipeline.InputParams") -> Image.Image:
        """
        - full: Keep existing flow (full img2img)
        - human: Use RVM to get soft alpha, generate full image, then replace only "human" region with alpha
        """
        # Update IP-Adapter scale if provided (real-time update)
        if hasattr(params, 'ip_adapter_scale') and self.stream.use_ip_adapter and self.stream.ip_adapter is not None:
            new_scale = float(params.ip_adapter_scale)
            old_scale = self.stream.ip_adapter.scale
            if abs(new_scale - old_scale) > 0.01:  # Only print when scale changes
                self.stream.ip_adapter.scale = new_scale
                self.stream.ip_adapter_scale = new_scale
                print(f"[IP-Adapter] Scale updated: {old_scale:.1f} → {new_scale:.1f}")
        
        # Ensure input image matches stream dimensions (width, height)
        frame_pil = params.image.convert("RGB").resize((self.stream.width, self.stream.height))

        # Existing img2img generation - calls _update_prompt_with_ip, using latest scale and ip_tokens
        image_tensor = self.stream.preprocess_image(frame_pil)
        gen_pil = self.stream(image=image_tensor, prompt=params.prompt)

        # Default full frame
        mode = getattr(params, "mode", "full")
        if mode != "human":
            return gen_pil

        # --- human only ---
        pha = self.masker.get_soft_mask(frame_pil)
        out = self.composite_human_only(frame_pil, gen_pil, pha)
        return out


    @staticmethod
    def composite_human_only(orig_pil: Image.Image, gen_pil: Image.Image, pha_torch) -> Image.Image:
        """
        Use RVM's soft alpha to replace only the "human" region with generated result:
        out = gen * alpha + orig * (1 - alpha)
        """
        alpha = pha_torch.numpy()[..., None]  # [H,W,1] 0~1
        f = np.array(orig_pil).astype(np.float32)
        g = np.array(gen_pil).astype(np.float32)
        out = g * alpha + f * (1.0 - alpha)
        return Image.fromarray(out.clip(0, 255).astype(np.uint8))
