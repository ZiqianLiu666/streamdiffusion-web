# rvm_masker.py
import torch
import torchvision.transforms.functional as TF

class RVMMasker:
    """
    Lightweight wrapper for PeterL1n/RobustVideoMatting
    - Maintains r1..r4 recurrent states for temporal stability
    - Returns soft alpha mask (0~1)
    """
    def __init__(self, device="cuda", model_name="resnet50", downsample_ratio=0.25):
        self.device = device
        self.model = torch.hub.load("PeterL1n/RobustVideoMatting", model_name).to(device).eval()
        self.r1 = self.r2 = self.r3 = self.r4 = None
        self.downsample_ratio = downsample_ratio

    @torch.inference_mode()
    def get_soft_mask(self, pil_image):
        # pil -> [1,3,H,W]
        src = TF.to_tensor(pil_image).unsqueeze(0).to(self.device)
        fgr, pha, self.r1, self.r2, self.r3, self.r4 = self.model(
            src, self.r1, self.r2, self.r3, self.r4, downsample_ratio=self.downsample_ratio
        )
        # pha: [1,1,H,W]  -> [H,W] 0~1
        return pha[0, 0].clamp(0, 1).cpu()

    def reset(self):
        self.r1 = self.r2 = self.r3 = self.r4 = None
