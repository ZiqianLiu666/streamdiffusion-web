# ip_adapter_sdturbo.py

from typing import Optional, Union

import torch
from torch import nn
from PIL import Image

from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor

import sys
import os
# Add SwiftEdit to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.ip_adapter.ip_adapter import ImageProjModel
from utils.ip_adapter.utils import is_torch2_available

if is_torch2_available():
    from utils.ip_adapter.attention_processor import (
        IPAttnProcessor2_0 as IPAttnProcessor,
        AttnProcessor2_0 as AttnProcessor,
    )
else:
    from utils.ip_adapter.attention_processor import IPAttnProcessor, AttnProcessor


class SDTurboIPAdapter(nn.Module):
    """
    IP-Adapter inference module for SD2.1 / SD-Turbo:
    - Replace unet.attn_processors with IPAttnProcessor / AttnProcessor
    - Load image_proj + ip_adapter weights from ip_adapter.bin
    - Provide set_reference_image() to get image_embeds and ip_tokens
    """

    def __init__(
        self,
        unet,                         # pipe.unet
        ip_adapter_ckpt: str,         # ip_adapter.bin
        image_encoder_path: str,      # CLIP image encoder path used during training
        device: str = "cuda",
        weight_dtype: torch.dtype = torch.float16,
        clip_extra_context_tokens: int = 4,
        is_sdxl: bool = False,
    ):
        super().__init__()
        self.unet = unet
        self.device = device
        self.weight_dtype = weight_dtype
        self.clip_extra_context_tokens = clip_extra_context_tokens
        
        
        # subfolder = "sdxl_models/image_encoder" if is_sdxl else "models/image_encoder"
        # # 1) CLIP image encoder + processor
        # self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        #     image_encoder_path,
        #     subfolder=subfolder,
        # ).to(device, dtype=weight_dtype)
        
        # If a specific directory with config.json is passed, load directly from that directory
        if os.path.isdir(image_encoder_path) and os.path.exists(os.path.join(image_encoder_path, "config.json")):
            encoder_dir = image_encoder_path
            encoder_subfolder = None
        else:
            # Otherwise assume it's a repo_id like "h94/IP-Adapter"
            encoder_dir = image_encoder_path
            encoder_subfolder = "sdxl_models/image_encoder" if is_sdxl else "models/image_encoder"

        load_kwargs = {}
        try:
            # New transformers supports low_cpu_mem_usage to save CPU peak memory
            load_kwargs = {"torch_dtype": weight_dtype, "low_cpu_mem_usage": True}
        except TypeError:
            # Older versions may not support this parameter, fall back to torch_dtype only
            load_kwargs = {"torch_dtype": weight_dtype}

        if encoder_subfolder is None:
            self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(
                encoder_dir,
                **load_kwargs,
            ).to(device)
        else:
            self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(
                encoder_dir,
                subfolder=encoder_subfolder,
                **load_kwargs,
            ).to(device)

        
        
        self.image_encoder.eval()
        self.image_encoder.requires_grad_(False)

        self.clip_image_processor = CLIPImageProcessor()

        # 2) image_proj_model: convert image_embeds -> extra tokens
        self.image_proj_model = ImageProjModel(
            cross_attention_dim=self.unet.config.cross_attention_dim,
            clip_embeddings_dim=self.image_encoder.config.projection_dim,
            clip_extra_context_tokens=self.clip_extra_context_tokens,
        ).to(device, dtype=weight_dtype)

        # 3) Initialize attention processors
        self._init_ip_attn_processors()

        # 4) Adapt to ModuleList for easy loading from checkpoint
        self.adapter_modules = nn.ModuleList(self.unet.attn_processors.values())

        # 5) Load ip_adapter.bin weights
        self._load_ip_adapter_weights(ip_adapter_ckpt)

        # 6) Runtime cache
        self.image_embeds: Optional[torch.Tensor] = None  # [1, C]
        self.ip_tokens: Optional[torch.Tensor] = None     # [1, K, D]
        self.scale: float = 1.0

    def _init_ip_attn_processors(self):
        """
        Same logic as in train_ip_s2_ldist_lnoise_v3.py:
        Determine hidden_size and cross_attention_dim based on name/block type,
        use IPAttnProcessor for cross-attn layers, AttnProcessor for self-attn.
        """
        attn_procs = {}
        unet_sd = self.unet.state_dict()

        for name in self.unet.attn_processors.keys():
            cross_attention_dim = None if name.endswith("attn1.processor") else self.unet.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = self.unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(self.unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = self.unet.config.block_out_channels[block_id]
            else:
                raise ValueError(f"unexpected attn processor name: {name}")

            if cross_attention_dim is None:
                attn_procs[name] = AttnProcessor()
            else:
                layer_name = name.split(".processor")[0]
                weights = {
                    "to_k_ip.weight": unet_sd[layer_name + ".to_k.weight"],
                    "to_v_ip.weight": unet_sd[layer_name + ".to_v.weight"],
                }
                proc = IPAttnProcessor(
                    hidden_size=hidden_size,
                    cross_attention_dim=cross_attention_dim,
                )
                proc.load_state_dict(weights)
                attn_procs[name] = proc

        self.unet.set_attn_processor(attn_procs)

    def _load_ip_adapter_weights(self, ckpt_path: str):
        state_dict = torch.load(ckpt_path, map_location="cpu")

        # 1) Case A: checkpoint is {"image_proj": ..., "ip_adapter": ...}
        if "image_proj" in state_dict and "ip_adapter" in state_dict:
            image_proj_sd = state_dict["image_proj"]
            adapter_sd = state_dict["ip_adapter"]
            print("[IP-Adapter] Detected dict-style ckpt with keys: 'image_proj' & 'ip_adapter'")

        else:
            # 2) Case B: checkpoint is full IPAdapter.state_dict()
            #    e.g., during stage-1 training, directly called state_dict() / load_state_dict() on IPAdapter(nn.Module)
            print("[IP-Adapter] Detected full IPAdapter state_dict, extracting submodules...")

            image_proj_sd = {
                k.replace("image_proj_model.", ""): v
                for k, v in state_dict.items()
                if k.startswith("image_proj_model.")
            }
            adapter_sd = {
                k.replace("adapter_modules.", ""): v
                for k, v in state_dict.items()
                if k.startswith("adapter_modules.")
            }

            if len(image_proj_sd) == 0 or len(adapter_sd) == 0:
                # If still empty here, checkpoint path/content is problematic
                raise KeyError(
                    "ip_adapter ckpt contains neither 'image_proj'/'ip_adapter' nor "
                    "'image_proj_model.*' / 'adapter_modules.*' keys. "
                    "Please check if ip_adapter.bin is from correct SwiftEdit/IP-Adapter training."
                )

        # Actually load weights
        self.image_proj_model.load_state_dict(image_proj_sd, strict=True)
        self.adapter_modules.load_state_dict(adapter_sd, strict=True)
        print(f"[IP-Adapter] Loaded weights from {ckpt_path}")


    @torch.no_grad()
    def set_reference_image(self, image: Union[str, Image.Image]):
        """
        Set / update reference image, generate image_embeds and ip_tokens.
        """
        if isinstance(image, str):
            pil_img = Image.open(image).convert("RGB")
        else:
            pil_img = image.convert("RGB")

        pixel = self.clip_image_processor(
            images=pil_img,
            return_tensors="pt",
        ).pixel_values.to(self.device, dtype=self.weight_dtype)

        self.image_embeds = self.image_encoder(pixel).image_embeds  # [1, C]
        self.ip_tokens = self.image_proj_model(self.image_embeds)   # [1, K, D]
        print(f"[IP-Adapter] ✅ Reference image set - ip_tokens shape: {self.ip_tokens.shape}, scale: {self.scale:.1f}")
