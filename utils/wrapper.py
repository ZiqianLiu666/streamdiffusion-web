import gc
import os
from pathlib import Path
import traceback
from typing import List, Literal, Optional, Union, Dict

import numpy as np
import torch
from diffusers import AutoencoderTiny, StableDiffusionPipeline, StableDiffusionXLPipeline
from PIL import Image

from huggingface_hub import hf_hub_download, snapshot_download

from streamdiffusion import StreamDiffusion
from streamdiffusion.image_utils import postprocess_image

from .ip_adapter_sdturbo import SDTurboIPAdapter



torch.set_grad_enabled(False)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


class StreamDiffusionWrapper:
    def __init__(
        self,
        model_id_or_path: str,
        t_index_list: List[int],
        lora_dict: Optional[Dict[str, float]] = None,
        mode: Literal["img2img", "txt2img"] = "img2img",
        output_type: Literal["pil", "pt", "np", "latent"] = "pil",
        lcm_lora_id: Optional[str] = None,
        vae_id: Optional[str] = None,
        device: Literal["cpu", "cuda"] = "cuda",
        dtype: torch.dtype = torch.float16,
        frame_buffer_size: int = 1,
        width: int = 512,
        height: int = 512,
        warmup: int = 10,
        acceleration: Literal["none", "xformers", "tensorrt"] = "tensorrt",
        do_add_noise: bool = True,
        device_ids: Optional[List[int]] = None,
        use_lcm_lora: bool = True,
        use_tiny_vae: bool = True,
        enable_similar_image_filter: bool = False,
        similar_image_filter_threshold: float = 0.98,
        similar_image_filter_max_skip_frame: int = 10,
        use_denoising_batch: bool = True,
        cfg_type: Literal["none", "full", "self", "initialize"] = "self",
        seed: int = 2,
        use_safety_checker: bool = False,
        engine_dir: Optional[Union[str, Path]] = "engines",
        
        # SD-Turbo-IPAdapter related configuration
        use_ip_adapter: bool = False,
        ip_adapter_ckpt: Optional[str] = None,        # ip_adapter.bin
        ip_image_encoder_path: Optional[str] = None,  # CLIP vision model path
        ip_ref_image: Optional[Union[str, Image.Image]] = None,
        ip_adapter_scale: float = 1.0,                # Reserved for scaling ip_tokens
        ip_adapter_repo_id: Optional[str] = None,
        ip_adapter_weight_name: Optional[str] = None,
        ip_image_encoder_repo_id: Optional[str] = None,
    ):
        """
        Initializes the StreamDiffusionWrapper.

        Parameters
        ----------
        model_id_or_path : str
            The model id or path to load.
        t_index_list : List[int]
            The t_index_list to use for inference.
        lora_dict : Optional[Dict[str, float]], optional
            The lora_dict to load, by default None.
            Keys are the LoRA names and values are the LoRA scales.
            Example: {'LoRA_1' : 0.5 , 'LoRA_2' : 0.7 ,...}
        mode : Literal["img2img", "txt2img"], optional
            txt2img or img2img, by default "img2img".
        output_type : Literal["pil", "pt", "np", "latent"], optional
            The output type of image, by default "pil".
        lcm_lora_id : Optional[str], optional
            The lcm_lora_id to load, by default None.
            If None, the default LCM-LoRA
            ("latent-consistency/lcm-lora-sdv1-5") will be used.
        vae_id : Optional[str], optional
            The vae_id to load, by default None.
            If None, the default TinyVAE
            ("madebyollin/taesd") will be used.
        device : Literal["cpu", "cuda"], optional
            The device to use for inference, by default "cuda".
        dtype : torch.dtype, optional
            The dtype for inference, by default torch.float16.
        frame_buffer_size : int, optional
            The frame buffer size for denoising batch, by default 1.
        width : int, optional
            The width of the image, by default 512.
        height : int, optional
            The height of the image, by default 512.
        warmup : int, optional
            The number of warmup steps to perform, by default 10.
        acceleration : Literal["none", "xformers", "tensorrt"], optional
            The acceleration method, by default "tensorrt".
        do_add_noise : bool, optional
            Whether to add noise for following denoising steps or not,
            by default True.
        device_ids : Optional[List[int]], optional
            The device ids to use for DataParallel, by default None.
        use_lcm_lora : bool, optional
            Whether to use LCM-LoRA or not, by default True.
        use_tiny_vae : bool, optional
            Whether to use TinyVAE or not, by default True.
        enable_similar_image_filter : bool, optional
            Whether to enable similar image filter or not,
            by default False.
        similar_image_filter_threshold : float, optional
            The threshold for similar image filter, by default 0.98.
        similar_image_filter_max_skip_frame : int, optional
            The max skip frame for similar image filter, by default 10.
        use_denoising_batch : bool, optional
            Whether to use denoising batch or not, by default True.
        cfg_type : Literal["none", "full", "self", "initialize"],
        optional
            The cfg_type for img2img mode, by default "self".
            You cannot use anything other than "none" for txt2img mode.
        seed : int, optional
            The seed, by default 2.
        use_safety_checker : bool, optional
            Whether to use safety checker or not, by default False.
        """
        # Check if it's an SDXL model
        self.is_sdxl = "sdxl" in model_id_or_path.lower()
        # Check if it's "2.x Turbo" (prevent sdxl-turbo from being treated as sd_turbo)
        self.sd_turbo = ("turbo" in model_id_or_path.lower()) and (not self.is_sdxl)
        
        self.ip_adapter_repo_id = ip_adapter_repo_id
        self.ip_adapter_weight_name = ip_adapter_weight_name
        self.ip_image_encoder_repo_id = ip_image_encoder_repo_id


        # IP-Adapter switch and required parameters
        self.use_ip_adapter = use_ip_adapter
        self.ip_adapter_ckpt = ip_adapter_ckpt
        self.ip_image_encoder_path = ip_image_encoder_path
        self.ip_ref_image = ip_ref_image
        self.ip_adapter_scale = ip_adapter_scale

        # Runtime objects
        self.ip_adapter: Optional[SDTurboIPAdapter] = None
        self.pipe = None  # Important: used by _update_prompt_with_ip
        
        if mode == "txt2img":
            if cfg_type != "none":
                raise ValueError(
                    f"txt2img mode accepts only cfg_type = 'none', but got {cfg_type}"
                )
            if use_denoising_batch and frame_buffer_size > 1:
                if not self.sd_turbo:
                    raise ValueError(
                        "txt2img mode cannot use denoising batch with frame_buffer_size > 1."
                    )

        if mode == "img2img":
            if not use_denoising_batch:
                raise NotImplementedError(
                    "img2img mode must use denoising batch for now."
                )

        self.device = device
        self.dtype = dtype
        self.width = width
        self.height = height
        self.mode = mode
        self.output_type = output_type
        self.frame_buffer_size = frame_buffer_size
        self.batch_size = (
            len(t_index_list) * frame_buffer_size
            if use_denoising_batch
            else frame_buffer_size
        )

        self.use_denoising_batch = use_denoising_batch
        self.use_safety_checker = use_safety_checker

        self.stream: StreamDiffusion = self._load_model(
            model_id_or_path=model_id_or_path,
            lora_dict=lora_dict,
            lcm_lora_id=lcm_lora_id,
            vae_id=vae_id,
            t_index_list=t_index_list,
            acceleration=acceleration,
            warmup=warmup,
            do_add_noise=do_add_noise,
            use_lcm_lora=use_lcm_lora,
            use_tiny_vae=use_tiny_vae,
            cfg_type=cfg_type,
            seed=seed,
            engine_dir=engine_dir,
        )

        if device_ids is not None:
            self.stream.unet = torch.nn.DataParallel(
                self.stream.unet, device_ids=device_ids
            )

        if enable_similar_image_filter:
            self.stream.enable_similar_image_filter(similar_image_filter_threshold, similar_image_filter_max_skip_frame)

    def prepare(
        self,
        prompt: str,
        negative_prompt: str = "",
        num_inference_steps: int = 50,
        guidance_scale: float = 1.2,
        delta: float = 1.0,
    ) -> None:
        """
        Prepares the model for inference.

        Parameters
        ----------
        prompt : str
            The prompt to generate images from.
        num_inference_steps : int, optional
            The number of inference steps to perform, by default 50.
        guidance_scale : float, optional
            The guidance scale to use, by default 1.2.
        delta : float, optional
            The delta multiplier of virtual residual noise,
            by default 1.0.
        """
        self.stream.prepare(
            prompt,
            negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            delta=delta,
        )

    def __call__(
        self,
        image: Optional[Union[str, Image.Image, torch.Tensor]] = None,
        prompt: Optional[str] = None,
    ) -> Union[Image.Image, List[Image.Image]]:
        """
        Performs img2img or txt2img based on the mode.

        Parameters
        ----------
        image : Optional[Union[str, Image.Image, torch.Tensor]]
            The image to generate from.
        prompt : Optional[str]
            The prompt to generate images from.

        Returns
        -------
        Union[Image.Image, List[Image.Image]]
            The generated image.
        """
        if self.mode == "img2img":
            return self.img2img(image, prompt)
        else:
            return self.txt2img(prompt)

    def txt2img(
        self, prompt: Optional[str] = None
    ) -> Union[Image.Image, List[Image.Image], torch.Tensor, np.ndarray]:
        """
        Performs txt2img.

        Parameters
        ----------
        prompt : Optional[str]
            The prompt to generate images from.

        Returns
        -------
        Union[Image.Image, List[Image.Image]]
            The generated image.
        """
        if prompt is not None:
            self.stream.update_prompt(prompt)

        if self.sd_turbo:
            image_tensor = self.stream.txt2img_sd_turbo(self.batch_size)
        else:
            image_tensor = self.stream.txt2img(self.frame_buffer_size)
        image = self.postprocess_image(image_tensor, output_type=self.output_type)

        if self.use_safety_checker:
            safety_checker_input = self.feature_extractor(
                image, return_tensors="pt"
            ).to(self.device)
            _, has_nsfw_concept = self.safety_checker(
                images=image_tensor.to(self.dtype),
                clip_input=safety_checker_input.pixel_values.to(self.dtype),
            )
            image = self.nsfw_fallback_img if has_nsfw_concept[0] else image

        return image

    def img2img(
        self, image: Union[str, Image.Image, torch.Tensor], prompt: Optional[str] = None
    ) -> Union[Image.Image, List[Image.Image], torch.Tensor, np.ndarray]:
        """
        Performs img2img.

        Parameters
        ----------
        image : Union[str, Image.Image, torch.Tensor]
            The image to generate from.

        Returns
        -------
        Image.Image
            The generated image.
        """
        if prompt is not None:
            if self.use_ip_adapter and self.ip_adapter is not None:
                # Use our own update logic
                self._update_prompt_with_ip(prompt)
            else:
                # Maintain original behavior
                self.stream.update_prompt(prompt)

        if isinstance(image, str) or isinstance(image, Image.Image):
            image = self.preprocess_image(image)

        image_tensor = self.stream(image)
        image = self.postprocess_image(image_tensor, output_type=self.output_type)

        if self.use_safety_checker:
            safety_checker_input = self.feature_extractor(
                image, return_tensors="pt"
            ).to(self.device)
            _, has_nsfw_concept = self.safety_checker(
                images=image_tensor.to(self.dtype),
                clip_input=safety_checker_input.pixel_values.to(self.dtype),
            )
            image = self.nsfw_fallback_img if has_nsfw_concept[0] else image

        return image

    def preprocess_image(self, image: Union[str, Image.Image]) -> torch.Tensor:
        """
        Preprocesses the image.

        Parameters
        ----------
        image : Union[str, Image.Image, torch.Tensor]
            The image to preprocess.

        Returns
        -------
        torch.Tensor
            The preprocessed image.
        """
        if isinstance(image, str):
            image = Image.open(image).convert("RGB").resize((self.width, self.height))
        if isinstance(image, Image.Image):
            image = image.convert("RGB").resize((self.width, self.height))

        return self.stream.image_processor.preprocess(
            image, self.height, self.width
        ).to(device=self.device, dtype=self.dtype)

    def postprocess_image(
        self, image_tensor: torch.Tensor, output_type: str = "pil"
    ) -> Union[Image.Image, List[Image.Image], torch.Tensor, np.ndarray]:
        """
        Postprocesses the image.

        Parameters
        ----------
        image_tensor : torch.Tensor
            The image tensor to postprocess.

        Returns
        -------
        Union[Image.Image, List[Image.Image]]
            The postprocessed image.
        """
        if self.frame_buffer_size > 1:
            return postprocess_image(image_tensor.cpu(), output_type=output_type)
        else:
            return postprocess_image(image_tensor.cpu(), output_type=output_type)[0]


    def set_ip_adapter_image(self, image: Union[str, Image.Image, None]):
        if not self.use_ip_adapter or self.ip_adapter is None:
            print("[IP-Adapter] not enabled, ignore.")
            return

        if image is None:
            self.ip_adapter.image_embeds = None
            self.ip_adapter.ip_tokens = None
            return

        self.ip_adapter.set_reference_image(image)



    @torch.no_grad()
    def _update_prompt_with_ip(self, prompt: str):
        """
        Instead of calling stream.update_prompt:
        1) Use pipe.encode_prompt to get text embedding
        2) If ip_tokens exist, concatenate them as in training script
        3) Expand to batch_size and write to stream.prompt_embeds
        """
        # 1) Text embedding - following update_prompt pattern
        encoder_output = self.pipe.encode_prompt(
            prompt=prompt,
            device=self.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=False,
        )
        # encoder_output[0] is prompt_embeds: [1, L, D]
        prompt_embeds = encoder_output[0]
        
        # SDXL support: save pooled_prompt_embeds (for added_cond_kwargs)
        if self.is_sdxl and len(encoder_output) > 1:
            # SDXL encode_prompt returns (prompt_embeds, pooled_prompt_embeds)
            pooled_prompt_embeds = encoder_output[1]  # [1, projection_dim] or None
            # If pooled_prompt_embeds is None, create a default value
            if pooled_prompt_embeds is None:
                # Get projection_dim from text_encoder_2
                if hasattr(self.pipe, 'text_encoder_2') and hasattr(self.pipe.text_encoder_2.config, 'projection_dim'):
                    projection_dim = self.pipe.text_encoder_2.config.projection_dim
                else:
                    projection_dim = 1280  # SDXL default
                # Create zero vector as fallback
                pooled_prompt_embeds = torch.zeros((1, projection_dim), device=self.device, dtype=self.dtype)
            # Save to stream for UNet forward
            self.stream.pooled_prompt_embeds = pooled_prompt_embeds.repeat(self.batch_size, 1)
        elif self.is_sdxl:
            # If encode_prompt didn't return pooled_prompt_embeds, create a default value
            if hasattr(self.pipe, 'text_encoder_2') and hasattr(self.pipe.text_encoder_2.config, 'projection_dim'):
                projection_dim = self.pipe.text_encoder_2.config.projection_dim
            else:
                projection_dim = 1280  # SDXL default
            pooled_prompt_embeds = torch.zeros((1, projection_dim), device=self.device, dtype=self.dtype)
            self.stream.pooled_prompt_embeds = pooled_prompt_embeds.repeat(self.batch_size, 1)

        # 2) If IP-Adapter exists, concatenate image tokens
        if self.ip_adapter is not None and self.ip_adapter.ip_tokens is not None:
            ip_tokens = self.ip_adapter.ip_tokens  # [1, K, D]
            # Apply scale (real-time update)
            current_scale = float(self.ip_adapter.scale)
            ip_tokens_scaled = ip_tokens * current_scale

            # If batch_size > 1 (denoising_batch/frame_buffer), concatenate in seq dimension first, then repeat
            prompt_embeds = torch.cat([prompt_embeds, ip_tokens_scaled], dim=1)  # [1, L+K, D]

        # 3) Expand according to stream's actual batch_size
        # Ensure stream.prompt_embeds is initialized to get correct batch_size
        if not hasattr(self.stream, 'prompt_embeds') or self.stream.prompt_embeds is None:
            # If prompt_embeds doesn't exist yet, call update_prompt once to initialize
            # This ensures StreamDiffusion's internal batch_size is set correctly
            self.stream.update_prompt(prompt)
        
        actual_batch_size = self.stream.prompt_embeds.shape[0]
        prompt_embeds = prompt_embeds.repeat(actual_batch_size, 1, 1)
        
        # SDXL support: also update pooled_prompt_embeds batch_size
        if self.is_sdxl and hasattr(self.stream, 'pooled_prompt_embeds') and self.stream.pooled_prompt_embeds is not None:
            # Ensure batch_size matches
            if self.stream.pooled_prompt_embeds.shape[0] != actual_batch_size:
                self.stream.pooled_prompt_embeds = self.stream.pooled_prompt_embeds[:1].repeat(actual_batch_size, 1)

        # 4) Directly write to stream.prompt_embeds (don't call update_prompt)
        self.stream.prompt_embeds = prompt_embeds



    def _load_model(
        self,
        model_id_or_path: str,
        t_index_list: List[int],
        lora_dict: Optional[Dict[str, float]] = None,
        lcm_lora_id: Optional[str] = None,
        vae_id: Optional[str] = None,
        acceleration: Literal["none", "xformers", "tensorrt"] = "tensorrt",
        warmup: int = 10,
        do_add_noise: bool = True,
        use_lcm_lora: bool = True,
        use_tiny_vae: bool = True,
        cfg_type: Literal["none", "full", "self", "initialize"] = "self",
        seed: int = 2,
        engine_dir: Optional[Union[str, Path]] = "engines",
    ) -> StreamDiffusion:
        """
        Loads the model.

        This method does the following:

        1. Loads the model from the model_id_or_path.
        2. Loads and fuses the LCM-LoRA model from the lcm_lora_id if needed.
        3. Loads the VAE model from the vae_id if needed.
        4. Enables acceleration if needed.
        5. Prepares the model for inference.
        6. Load the safety checker if needed.

        Parameters
        ----------
        model_id_or_path : str
            The model id or path to load.
        t_index_list : List[int]
            The t_index_list to use for inference.
        lora_dict : Optional[Dict[str, float]], optional
            The lora_dict to load, by default None.
            Keys are the LoRA names and values are the LoRA scales.
            Example: {'LoRA_1' : 0.5 , 'LoRA_2' : 0.7 ,...}
        lcm_lora_id : Optional[str], optional
            The lcm_lora_id to load, by default None.
        vae_id : Optional[str], optional
            The vae_id to load, by default None.
        acceleration : Literal["none", "xfomers", "sfast", "tensorrt"], optional
            The acceleration method, by default "tensorrt".
        warmup : int, optional
            The number of warmup steps to perform, by default 10.
        do_add_noise : bool, optional
            Whether to add noise for following denoising steps or not,
            by default True.
        use_lcm_lora : bool, optional
            Whether to use LCM-LoRA or not, by default True.
        use_tiny_vae : bool, optional
            Whether to use TinyVAE or not, by default True.
        cfg_type : Literal["none", "full", "self", "initialize"],
        optional
            The cfg_type for img2img mode, by default "self".
            You cannot use anything other than "none" for txt2img mode.
        seed : int, optional
            The seed, by default 2.

        Returns
        -------
        StreamDiffusion
            The loaded model.
        """

        # ---------- 1. Load Diffusers Pipeline ----------
        try:
            if self.is_sdxl:
                # SDXL-Turbo: use dedicated XL Pipeline with low_cpu_mem_usage to reduce CPU memory peak
                print("[Pipeline] Loading SDXL-Turbo model with low CPU memory usage...")
                pipe: StableDiffusionXLPipeline = StableDiffusionXLPipeline.from_pretrained(
                    model_id_or_path,
                    torch_dtype=self.dtype,
                    low_cpu_mem_usage=True,  # Key: reduce CPU memory peak
                    variant="fp16" if self.dtype == torch.float16 else None,  # Directly load FP16 variant
                )
                # Move to GPU step by step to avoid occupying too much memory at once
                pipe = pipe.to(self.device)
                # Clear CPU cache
                gc.collect()
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
            else:
                # SD1.x / SD2.x / SD-Turbo: use standard SD Pipeline
                pipe: StableDiffusionPipeline = StableDiffusionPipeline.from_pretrained(
                    model_id_or_path,
                    torch_dtype=self.dtype,
                    low_cpu_mem_usage=True,  # Also add this optimization
                    
                ).to(self.device)
        except Exception:
            traceback.print_exc()
            print("Model load has failed. Doesn't exist.")
            exit()

        self.pipe = pipe  # Used by encode_prompt later

        # ---------- 2. Prepare IP-Adapter weights and image encoder local paths ----------
        ckpt_path = self.ip_adapter_ckpt  # Allow manual local path

        if self.use_ip_adapter:
            # 2.1 adapter weights: local first, otherwise download from HF once
            if ckpt_path is None:
                if self.ip_adapter_repo_id is None or self.ip_adapter_weight_name is None:
                    raise ValueError("[IP-Adapter] Neither local ip_adapter_ckpt nor repo_id/weight_name provided")
                print(f"[HF] Downloading IP-Adapter weights: repo={self.ip_adapter_repo_id}, file={self.ip_adapter_weight_name}")
                ckpt_path = hf_hub_download(
                    repo_id=self.ip_adapter_repo_id,
                    filename=self.ip_adapter_weight_name,
                )
                # hf_hub_download uses local cache, second run won't use network, returns path instantly
            else:
                print(f"[IP-Adapter] Using local IP-Adapter ckpt: {ckpt_path}")

            # 2.2 image encoder: use snapshot_download to download files from specified subdirectory to local
            if self.ip_image_encoder_repo_id is not None:
                encoder_repo = self.ip_image_encoder_repo_id
            else:
                encoder_repo = self.ip_adapter_repo_id  # Fallback

            if encoder_repo is not None:
                # Select correct image encoder subdirectory based on model type
                if self.is_sdxl:
                    encoder_subdir = "sdxl_models/image_encoder"  # SDXL
                else:
                    encoder_subdir = "models/image_encoder"  # SD1.5/SD2.1
                print(f"[HF] Downloading IP-Adapter image encoder from repo={encoder_repo}, subdir={encoder_subdir}")
                encoder_root = snapshot_download(
                    repo_id=encoder_repo,
                    allow_patterns=[f"{encoder_subdir}/*"],
                )
                # Result format: ~/.cache/huggingface/hub/models--h94--IP-Adapter/snapshots/<hash>/
                self.ip_image_encoder_path = os.path.join(encoder_root, encoder_subdir)
                print(f"[HF] Local image encoder dir: {self.ip_image_encoder_path}")

        # ---------- 3. Actually initialize IP-Adapter ----------
        if self.use_ip_adapter and ckpt_path is not None and self.ip_image_encoder_path is not None:
            print("[IP-Adapter] Initializing generic IP-Adapter for UNet ...")
            self.ip_adapter = SDTurboIPAdapter(
                unet=pipe.unet,
                ip_adapter_ckpt=ckpt_path,
                image_encoder_path=self.ip_image_encoder_path,  # Note: pass directory here
                device=self.device,
                weight_dtype=self.dtype,
                clip_extra_context_tokens=4,
                is_sdxl=self.is_sdxl,  # Currently SD-Turbo; set True if using SDXL later
            ).to(self.device)

            if self.ip_ref_image is not None:
                self.ip_adapter.set_reference_image(self.ip_ref_image)
                self.ip_adapter.scale = self.ip_adapter_scale




        stream = StreamDiffusion(
            pipe=pipe,
            t_index_list=t_index_list,
            torch_dtype=self.dtype,
            width=self.width,
            height=self.height,
            do_add_noise=do_add_noise,
            frame_buffer_size=self.frame_buffer_size,
            use_denoising_batch=self.use_denoising_batch,
            cfg_type=cfg_type,
        )
        
        # Pass ip_tokens to stream (will be used in StreamDiffusion later)
        if self.ip_adapter is not None:
            stream.ip_adapter = self.ip_adapter     # Attach as attribute
            stream.ip_tokens = self.ip_adapter.ip_tokens  # Convenient direct access

        
        # SDXL support: Monkey patch UNet forward to automatically add added_cond_kwargs
        if self.is_sdxl:
            original_unet_forward = stream.unet.forward
            
            def patched_unet_forward(sample, timestep, encoder_hidden_states, **kwargs):
                """Automatically add added_cond_kwargs for SDXL"""
                # If added_cond_kwargs already provided, use directly
                if "added_cond_kwargs" in kwargs and kwargs["added_cond_kwargs"] is not None:
                    return original_unet_forward(sample, timestep, encoder_hidden_states, **kwargs)
                
                # Otherwise, auto-generate added_cond_kwargs
                batch_size = sample.shape[0]
                device = sample.device
                dtype = sample.dtype
                
                # Generate time_ids (required by SDXL)
                # time_ids format: [original_size, crops_coords_top_left, target_size]
                original_size = (self.height, self.width)
                crops_coords_top_left = (0, 0)
                target_size = (self.height, self.width)
                
                # Build time_ids: [batch_size, 6]
                time_ids = torch.tensor([
                    original_size[0], original_size[1],
                    crops_coords_top_left[0], crops_coords_top_left[1],
                    target_size[0], target_size[1]
                ], device=device, dtype=dtype).repeat(batch_size, 1)
                
                # Get text_embeds (from text_encoder_2's pooled output)
                # For SDXL-Turbo, use zero vector if pooled_prompt_embeds not available
                if hasattr(stream, 'pooled_prompt_embeds') and stream.pooled_prompt_embeds is not None:
                    text_embeds = stream.pooled_prompt_embeds
                    # Ensure batch_size matches
                    if text_embeds.shape[0] != batch_size:
                        text_embeds = text_embeds[:1].repeat(batch_size, 1)
                else:
                    # Create zero vector (SDXL-Turbo may not need it, but for compatibility)
                    projection_dim = getattr(pipe.text_encoder_2.config, 'projection_dim', 1280) if hasattr(pipe, 'text_encoder_2') else 1280
                    text_embeds = torch.zeros((batch_size, projection_dim), device=device, dtype=dtype)
                
                # Build added_cond_kwargs
                kwargs["added_cond_kwargs"] = {
                    "text_embeds": text_embeds,
                    "time_ids": time_ids,
                }
                
                return original_unet_forward(sample, timestep, encoder_hidden_states, **kwargs)
            
            stream.unet.forward = patched_unet_forward
            print("[SDXL] ✅ Patched UNet forward to support added_cond_kwargs")
        
        if not self.sd_turbo:
            if use_lcm_lora:
                if lcm_lora_id is not None:
                    stream.load_lcm_lora(
                        pretrained_model_name_or_path_or_dict=lcm_lora_id
                    )
                else:
                    stream.load_lcm_lora()
                stream.fuse_lora()

            if lora_dict is not None:
                for lora_name, lora_scale in lora_dict.items():
                    stream.load_lora(lora_name)
                    stream.fuse_lora(lora_scale=lora_scale)
                    print(f"Use LoRA: {lora_name} in weights {lora_scale}")

        if use_tiny_vae:
            if vae_id is not None:
                # Use provided vae_id (already selected based on model type in img2img.py)
                stream.vae = AutoencoderTiny.from_pretrained(vae_id).to(
                    device=pipe.device, dtype=pipe.dtype
                )
            else:
                # If vae_id not specified, auto-select based on model type
                if self.is_sdxl:
                    default_vae = "madebyollin/taesdxl"
                else:
                    default_vae = "madebyollin/taesd"
                stream.vae = AutoencoderTiny.from_pretrained(default_vae).to(
                    device=pipe.device, dtype=pipe.dtype
                )

        try:
            if acceleration == "xformers":
                stream.pipe.enable_xformers_memory_efficient_attention()
            if acceleration == "tensorrt":
                from polygraphy import cuda
                from streamdiffusion.acceleration.tensorrt import (
                    TorchVAEEncoder,
                    compile_unet,
                    compile_vae_decoder,
                    compile_vae_encoder,
                )
                from streamdiffusion.acceleration.tensorrt.engine import (
                    AutoencoderKLEngine,
                    UNet2DConditionModelEngine,
                )
                from streamdiffusion.acceleration.tensorrt.models import (
                    VAE,
                    UNet,
                    VAEEncoder,
                )

                def create_prefix(
                    model_id_or_path: str,
                    max_batch_size: int,
                    min_batch_size: int,
                ):
                    maybe_path = Path(model_id_or_path)
                    if maybe_path.exists():
                        return f"{maybe_path.stem}--lcm_lora-{use_lcm_lora}--tiny_vae-{use_tiny_vae}--max_batch-{max_batch_size}--min_batch-{min_batch_size}--mode-{self.mode}"
                    else:
                        return f"{model_id_or_path}--lcm_lora-{use_lcm_lora}--tiny_vae-{use_tiny_vae}--max_batch-{max_batch_size}--min_batch-{min_batch_size}--mode-{self.mode}"

                engine_dir = Path(engine_dir)
                unet_path = os.path.join(
                    engine_dir,
                    create_prefix(
                        model_id_or_path=model_id_or_path,
                        max_batch_size=stream.trt_unet_batch_size,
                        min_batch_size=stream.trt_unet_batch_size,
                    ),
                    "unet.engine",
                )
                vae_encoder_path = os.path.join(
                    engine_dir,
                    create_prefix(
                        model_id_or_path=model_id_or_path,
                        max_batch_size=self.batch_size
                        if self.mode == "txt2img"
                        else stream.frame_bff_size,
                        min_batch_size=self.batch_size
                        if self.mode == "txt2img"
                        else stream.frame_bff_size,
                    ),
                    "vae_encoder.engine",
                )
                vae_decoder_path = os.path.join(
                    engine_dir,
                    create_prefix(
                        model_id_or_path=model_id_or_path,
                        max_batch_size=self.batch_size
                        if self.mode == "txt2img"
                        else stream.frame_bff_size,
                        min_batch_size=self.batch_size
                        if self.mode == "txt2img"
                        else stream.frame_bff_size,
                    ),
                    "vae_decoder.engine",
                )

                if not os.path.exists(unet_path):
                    os.makedirs(os.path.dirname(unet_path), exist_ok=True)
                    unet_model = UNet(
                        fp16=True,
                        device=stream.device,
                        max_batch_size=stream.trt_unet_batch_size,
                        min_batch_size=stream.trt_unet_batch_size,
                        embedding_dim=stream.text_encoder.config.hidden_size,
                        unet_dim=stream.unet.config.in_channels,
                    )
                    compile_unet(
                        stream.unet,
                        unet_model,
                        unet_path + ".onnx",
                        unet_path + ".opt.onnx",
                        unet_path,
                        opt_batch_size=stream.trt_unet_batch_size,
                    )

                if not os.path.exists(vae_decoder_path):
                    os.makedirs(os.path.dirname(vae_decoder_path), exist_ok=True)
                    stream.vae.forward = stream.vae.decode
                    vae_decoder_model = VAE(
                        device=stream.device,
                        max_batch_size=self.batch_size
                        if self.mode == "txt2img"
                        else stream.frame_bff_size,
                        min_batch_size=self.batch_size
                        if self.mode == "txt2img"
                        else stream.frame_bff_size,
                    )
                    compile_vae_decoder(
                        stream.vae,
                        vae_decoder_model,
                        vae_decoder_path + ".onnx",
                        vae_decoder_path + ".opt.onnx",
                        vae_decoder_path,
                        opt_batch_size=self.batch_size
                        if self.mode == "txt2img"
                        else stream.frame_bff_size,
                    )
                    delattr(stream.vae, "forward")

                if not os.path.exists(vae_encoder_path):
                    os.makedirs(os.path.dirname(vae_encoder_path), exist_ok=True)
                    vae_encoder = TorchVAEEncoder(stream.vae).to(torch.device("cuda"))
                    vae_encoder_model = VAEEncoder(
                        device=stream.device,
                        max_batch_size=self.batch_size
                        if self.mode == "txt2img"
                        else stream.frame_bff_size,
                        min_batch_size=self.batch_size
                        if self.mode == "txt2img"
                        else stream.frame_bff_size,
                    )
                    compile_vae_encoder(
                        vae_encoder,
                        vae_encoder_model,
                        vae_encoder_path + ".onnx",
                        vae_encoder_path + ".opt.onnx",
                        vae_encoder_path,
                        opt_batch_size=self.batch_size
                        if self.mode == "txt2img"
                        else stream.frame_bff_size,
                    )

                cuda_stream = cuda.Stream()

                vae_config = stream.vae.config
                vae_dtype = stream.vae.dtype

                stream.unet = UNet2DConditionModelEngine(
                    unet_path, cuda_stream, use_cuda_graph=False
                )
                stream.vae = AutoencoderKLEngine(
                    vae_encoder_path,
                    vae_decoder_path,
                    cuda_stream,
                    stream.pipe.vae_scale_factor,
                    use_cuda_graph=False,
                )
                setattr(stream.vae, "config", vae_config)
                setattr(stream.vae, "dtype", vae_dtype)

                gc.collect()
                torch.cuda.empty_cache()

                print("TensorRT acceleration enabled.")
            if acceleration == "sfast":
                from streamdiffusion.acceleration.sfast import (
                    accelerate_with_stable_fast,
                )

                stream = accelerate_with_stable_fast(stream)
                print("StableFast acceleration enabled.")
        except Exception:
            traceback.print_exc()
            print("Acceleration has failed. Falling back to normal mode.")

        if seed < 0: # Random seed
            seed = np.random.randint(0, 1000000)

        stream.prepare(
            "",
            "",
            num_inference_steps=50,
            guidance_scale=1.1
            if stream.cfg_type in ["full", "self", "initialize"]
            else 1.0,
            generator=torch.manual_seed(seed),
            seed=seed,
        )

        if self.use_safety_checker:
            from transformers import CLIPFeatureExtractor
            from diffusers.pipelines.stable_diffusion.safety_checker import (
                StableDiffusionSafetyChecker,
            )

            self.safety_checker = StableDiffusionSafetyChecker.from_pretrained(
                "CompVis/stable-diffusion-safety-checker"
            ).to(pipe.device)
            self.feature_extractor = CLIPFeatureExtractor.from_pretrained(
                "openai/clip-vit-base-patch32"
            )
            self.nsfw_fallback_img = Image.new("RGB", (512, 512), (0, 0, 0))

        return stream
