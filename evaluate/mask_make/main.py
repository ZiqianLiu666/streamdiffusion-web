#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Evaluate three matting / segmentation methods on withoutBG100:

1. RVM (Robust Video Matting) - alpha prediction
2. GrabCut (traditional CV baseline) - binary mask
3. YOLOv8 + SAM - detection + prompted segmentation

Metrics:
- Matting metrics (on alpha, following matting literature):
    * SAD  (Sum of Absolute Differences)
    * MSE  (Mean Squared Error)
    * Grad (Gradient error, Sobel-based approximation)
    * Conn (Connectivity error, Rhemann-style approximation)

- Segmentation metrics (on thresholded masks):
    * IoU
    * Dice
    * Precision
    * Recall

Also measure:
- Average runtime per image (seconds)
- FPS = 1 / avg_time

Usage example:
python eval_withoutbg100.py \
    --data-root /path/to/withoutbg100 \
    --images-subdir images \
    --alpha-subdir alpha \
    --device cuda \
    --yolo-weights yolov8s.pt \
    --sam-checkpoint sam_vit_b.pth \
    --sam-model-type vit_b \
    --output-csv results_withoutbg100.csv
"""

import os
import glob
import time
import argparse
from typing import Tuple, Dict, List

import numpy as np
from PIL import Image
import cv2

import torch
from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor

import requests
from tqdm import tqdm


# =========================
# Utility: auto download
# =========================

def download_with_progress(url: str, save_path: str, chunk_size: int = 8192):
    """
    Download a file from `url` to `save_path` with a tqdm progress bar.
    If the file already exists, skip download.
    """
    if os.path.exists(save_path):
        print(f"[download] File already exists, skip: {save_path}")
        return

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    print(f"[download] Downloading from {url}")
    r = requests.get(url, stream=True)
    r.raise_for_status()
    total = int(r.headers.get("content-length", 0))

    with open(save_path, "wb") as f, tqdm(
        total=total, unit="B", unit_scale=True, desc=os.path.basename(save_path)
    ) as pbar:
        for chunk in r.iter_content(chunk_size=chunk_size):
            if chunk:  # filter out keep-alive chunks
                f.write(chunk)
                pbar.update(len(chunk))

    print(f"[download] Saved to {save_path}")


def ensure_sam_checkpoint(checkpoint_path: str, model_type: str = "vit_b"):
    """
    Ensure SAM checkpoint exists locally.
    If not, download it from Meta official URL (for the given model_type).

    model_type: vit_b / vit_l / vit_h
    """
    if os.path.exists(checkpoint_path):
        print(f"[SAM] Using existing checkpoint: {checkpoint_path}")
        return

    # Official weight URLs (Meta AI Segment Anything)
    # Common ones: vit_b / vit_l / vit_h
    sam_urls = {
        "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
        "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
        "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    }

    if model_type not in sam_urls:
        raise ValueError(f"Unknown SAM model_type: {model_type}, "
                         f"available: {list(sam_urls.keys())}")

    url = sam_urls[model_type]
    print(f"[SAM] Checkpoint not found. Will download {model_type} from:\n  {url}")
    download_with_progress(url, checkpoint_path)


# =========================
# Data loading
# =========================

def load_image_alpha_pair(
    img_path: str,
    alpha_path: str,
    target_size: Tuple[int, int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load an RGB image and its alpha matte, return:
        img:  H x W x 3, float32, [0, 1], RGB
        alpha: H x W, float32, [0, 1]
    Optionally resize to target_size (W, H).
    """
    img = Image.open(img_path).convert("RGB")
    alpha_img = Image.open(alpha_path)

    if target_size is not None:
        w, h = target_size
        img = img.resize((w, h), Image.BILINEAR)
        alpha_img = alpha_img.resize((w, h), Image.BILINEAR)

    img_np = np.array(img, dtype=np.uint8)
    alpha_np = np.array(alpha_img)

    # alpha may be single-channel or RGBA; try to robustly extract
    if alpha_np.ndim == 3:
        # If RGBA, take last channel; otherwise average
        if alpha_np.shape[2] == 4:
            alpha_np = alpha_np[:, :, 3]
        else:
            alpha_np = alpha_np.mean(axis=2)

    # Normalize to [0,1]
    if alpha_np.dtype != np.float32:
        alpha_np = alpha_np.astype(np.float32)
    alpha_np /= 255.0
    alpha_np = np.clip(alpha_np, 0.0, 1.0)

    img_np = img_np.astype(np.float32) / 255.0

    return img_np, alpha_np


# =========================
# Matting / segmentation metrics
# =========================

def compute_sad(pred: np.ndarray, gt: np.ndarray) -> float:
    """SAD: Sum of Absolute Differences between alpha mattes."""
    return float(np.abs(pred - gt).sum())


def compute_mse(pred: np.ndarray, gt: np.ndarray) -> float:
    """MSE: Mean Squared Error between alpha mattes."""
    return float(((pred - gt) ** 2).mean())


def compute_gradient_error(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    Gradient error (approximation):
    - Compute Sobel gradients for pred and gt
    - Take absolute difference of gradient magnitudes and sum
    """
    pred32 = pred.astype(np.float32)
    gt32 = gt.astype(np.float32)

    def sobel_grad(x: np.ndarray) -> np.ndarray:
        gx = cv2.Sobel(x, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(x, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.sqrt(gx ** 2 + gy ** 2)
        return mag

    gp = sobel_grad(pred32)
    gg = sobel_grad(gt32)

    return float(np.abs(gp - gg).sum())


def _compute_L_map(alpha: np.ndarray, step: float = 0.1) -> np.ndarray:
    """
    Approximate the L-map used in connectivity error definition.

    alpha: [H,W] float32 in [0,1]
    step: threshold step, e.g. 0.1

    Returns:
        L: [H,W] float32 in [0,1]
    """
    alpha = alpha.astype(np.float32)
    H, W = alpha.shape
    L = np.zeros_like(alpha, dtype=np.float32)
    visited = np.zeros_like(alpha, dtype=bool)

    thresholds = np.arange(0, 1.0, step)[1:]  # 0.1, 0.2, ..., 0.9

    for t in thresholds:
        mask = alpha >= t
        if not np.any(mask):
            continue

        mask_u8 = mask.astype(np.uint8)
        num_labels, labels = cv2.connectedComponents(mask_u8)

        if num_labels <= 1:
            continue

        # Find largest connected component (label 0 is background)
        max_area = 0
        max_label = 0
        for lab in range(1, num_labels):
            area = (labels == lab).sum()
            if area > max_area:
                max_area = area
                max_label = lab

        largest_cc = (labels == max_label)
        new_pixels = largest_cc & (~visited)
        L[new_pixels] = t
        visited |= largest_cc

    # Any alpha>0 that never got assigned gets L=1.0
    remaining = (alpha > 0) & (~visited)
    L[remaining] = 1.0

    return L


def compute_connectivity_error(pred: np.ndarray, gt: np.ndarray, step: float = 0.1) -> float:
    """
    Connectivity error (approximation of Rhemann connectivity metric):

    1) For each alpha, compute L-map with progressively growing largest
       connected component over thresholds.
    2) Error is mean absolute difference between L_pred and L_gt.
    """
    L_pred = _compute_L_map(pred, step)
    L_gt = _compute_L_map(gt, step)
    return float(np.abs(L_pred - L_gt).mean())


def compute_segmentation_metrics(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray
) -> Dict[str, float]:
    """
    pred_mask, gt_mask: bool or 0/1 arrays (H,W)

    Returns:
        dict with IoU, Dice, Precision, Recall
    """
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)

    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, np.logical_not(gt)).sum()
    fn = np.logical_and(np.logical_not(pred), gt).sum()

    union = np.logical_or(pred, gt).sum()

    iou = tp / (union + 1e-8)
    dice = 2 * tp / (pred.sum() + gt.sum() + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)

    return {
        "IoU": float(iou),
        "Dice": float(dice),
        "Precision": float(precision),
        "Recall": float(recall),
    }


# =========================
# Methods
# =========================

class RVMWrapper:
    """
    RVM model wrapper using TorchHub.

    NOTE: This is frame-independent usage; for real video you
    would maintain recurrent states across frames for best results.
    """

    def __init__(self, device: str = "cuda", variant: str = "mobilenetv3", downsample_ratio: float = 0.25):
        self.device = device
        self.downsample_ratio = downsample_ratio
        # TorchHub load
        self.model = torch.hub.load("PeterL1n/RobustVideoMatting", variant)
        self.model.eval().to(self.device)

    @torch.no_grad()
    def infer(self, img_np: np.ndarray) -> np.ndarray:
        """
        img_np: H x W x 3, float32, [0,1], RGB

        Returns:
            alpha: H x W, float32, [0,1]
        """
        h, w, _ = img_np.shape
        src = torch.from_numpy(img_np.transpose(2, 0, 1)).unsqueeze(0)  # 1x3xHxW
        src = src.to(self.device, dtype=torch.float32)

        # No temporal memory for single images
        rec = [None] * 4
        fgr, pha, *rec = self.model(src, *rec, self.downsample_ratio)

        alpha = pha[0, 0].detach().cpu().numpy()
        alpha = np.clip(alpha, 0.0, 1.0)

        return alpha


class GrabCutWrapper:
    """
    Traditional CV baseline: GrabCut.
    """

    def __init__(self, iter_count: int = 5, margin_ratio: float = 0.05):
        self.iter_count = iter_count
        self.margin_ratio = margin_ratio

    def infer(self, img_np: np.ndarray) -> np.ndarray:
        """
        img_np: H x W x 3, float32, [0,1], RGB

        Returns:
            mask: H x W, float32, [0,1]
        """
        img_u8 = (img_np * 255.0).astype(np.uint8)
        h, w, _ = img_u8.shape

        x = int(self.margin_ratio * w)
        y = int(self.margin_ratio * h)
        bw = int((1.0 - 2 * self.margin_ratio) * w)
        bh = int((1.0 - 2 * self.margin_ratio) * h)
        rect = (x, y, bw, bh)

        mask = np.zeros((h, w), np.uint8)
        bgModel = np.zeros((1, 65), np.float64)
        fgModel = np.zeros((1, 65), np.float64)

        cv2.grabCut(img_u8, mask, rect, bgModel, fgModel,
                    self.iter_count, cv2.GC_INIT_WITH_RECT)

        # 0,2: background; 1,3: foreground
        mask_bin = np.where((mask == 1) | (mask == 3), 1.0, 0.0).astype(np.float32)

        return mask_bin


class YoloSamWrapper:
    """
    YOLOv8 + SAM pipeline.

    - YOLOv8: detect objects and get bounding boxes
    - SAM: use boxes as prompts to generate masks
    - Combine masks (union) as foreground
    """

    def __init__(
        self,
        yolo_weights: str,
        sam_checkpoint: str,
        sam_model_type: str = "vit_b",
        device: str = "cuda",
        yolo_conf: float = 0.25
    ):
        self.device = device
        self.yolo_conf = yolo_conf

        # YOLO model (ultralytics will auto-download weights)
        self.yolo = YOLO(yolo_weights)
        self.yolo.to(device)

        # SAM model (checkpoint is guaranteed to exist by ensure_sam_checkpoint)
        sam = sam_model_registry[sam_model_type](checkpoint=sam_checkpoint)
        sam.to(device)
        sam.eval()
        self.sam_predictor = SamPredictor(sam)

    @torch.no_grad()
    def infer(self, img_np: np.ndarray) -> np.ndarray:
        """
        img_np: H x W x 3, float32, [0,1], RGB

        Returns:
            mask: H x W, float32, [0,1]
        """
        H, W, _ = img_np.shape

        img_u8 = (img_np * 255.0).astype(np.uint8)
        img_bgr = cv2.cvtColor(img_u8, cv2.COLOR_RGB2BGR)

        results = self.yolo.predict(
            img_bgr,
            conf=self.yolo_conf,
            verbose=False,
            device=self.device
        )

        if len(results) == 0:
            return np.zeros((H, W), dtype=np.float32)

        res = results[0]
        if res.boxes is None or res.boxes.xyxy is None or res.boxes.shape[0] == 0:
            return np.zeros((H, W), dtype=np.float32)

        boxes_xyxy = res.boxes.xyxy.cpu().numpy()  # Nx4

        # SAM segmentation (on RGB)
        self.sam_predictor.set_image(img_u8)  # expects RGB uint8

        final_mask = np.zeros((H, W), dtype=bool)

        for box in boxes_xyxy:
            box_input = box[None, :]  # 1x4
            masks, scores, logits = self.sam_predictor.predict(
                box=box_input,
                multimask_output=False
            )
            mask = masks[0].astype(bool)
            final_mask |= mask

        return final_mask.astype(np.float32)


# =========================
# Evaluation loop
# =========================

def evaluate_on_dataset(
    data_root: str,
    images_subdir: str,
    alpha_subdir: str,
    device: str = "cuda",
    yolo_weights: str = "yolov8s.pt",
    sam_checkpoint: str = "sam_vit_b.pth",
    sam_model_type: str = "vit_b",
    target_width: int = 512,
    target_height: int = 512,
    output_csv: str = None
):
    images_dir = os.path.join(data_root, images_subdir)
    alpha_dir = os.path.join(data_root, alpha_subdir)

    img_paths = sorted(glob.glob(os.path.join(images_dir, "*")))
    assert len(img_paths) > 0, f"No images found in {images_dir}"

    print(f"Found {len(img_paths)} images.")

    # Ensure SAM weights exist (auto-download if not)
    ensure_sam_checkpoint(sam_checkpoint, model_type=sam_model_type)

    # Initialize methods
    print("Initializing models...")
    rvm = RVMWrapper(device=device, variant="mobilenetv3", downsample_ratio=0.25)
    grabcut = GrabCutWrapper(iter_count=5, margin_ratio=0.05)
    yolo_sam = YoloSamWrapper(
        yolo_weights=yolo_weights,
        sam_checkpoint=sam_checkpoint,
        sam_model_type=sam_model_type,
        device=device,
        yolo_conf=0.25
    )

    methods = ["RVM", "GrabCut", "YOLO+SAM"]
    mask_metrics = ["IoU", "Dice"]  # Keep only two core mask metrics

    sums = {
        m: {
            "IoU": 0.0,
            "Dice": 0.0,
            "Time": 0.0,
        } for m in methods
    }


    results_per_image = []

    target_size = (target_width, target_height)

    for idx, img_path in enumerate(img_paths):
        basename = os.path.basename(img_path)
        name_no_ext, _ = os.path.splitext(basename)

        alpha_candidates = glob.glob(os.path.join(alpha_dir, name_no_ext + ".*"))
        if len(alpha_candidates) == 0:
            print(f"[Warning] No alpha found for {basename}, skipping.")
            continue
        alpha_path = alpha_candidates[0]

        img_np, alpha_gt = load_image_alpha_pair(
            img_path,
            alpha_path,
            target_size=target_size
        )

        gt_mask = alpha_gt > 0.5

        per_image_entry = {
            "name": name_no_ext,
        }

        # ---- RVM ----
        # ---- RVM ----
        t0 = time.perf_counter()
        alpha_rvm = rvm.infer(img_np)
        t1 = time.perf_counter()
        dt_rvm = t1 - t0

        # Threshold RVM's alpha into mask
        mask_rvm = alpha_rvm > 0.5
        seg_rvm = compute_segmentation_metrics(mask_rvm, gt_mask)

        sums["RVM"]["IoU"]  += seg_rvm["IoU"]
        sums["RVM"]["Dice"] += seg_rvm["Dice"]
        sums["RVM"]["Time"] += dt_rvm

        per_image_entry.update({
            "RVM_IoU":  seg_rvm["IoU"],
            "RVM_Dice": seg_rvm["Dice"],
            "RVM_Time": dt_rvm,
        })


        # ---- GrabCut ----
        # ---- GrabCut ----
        t0 = time.perf_counter()
        mask_gc = grabcut.infer(img_np)
        t1 = time.perf_counter()
        dt_gc = t1 - t0

        mask_gc_bin = mask_gc > 0.5
        seg_gc = compute_segmentation_metrics(mask_gc_bin, gt_mask)

        sums["GrabCut"]["IoU"]  += seg_gc["IoU"]
        sums["GrabCut"]["Dice"] += seg_gc["Dice"]
        sums["GrabCut"]["Time"] += dt_gc

        per_image_entry.update({
            "GC_IoU":  seg_gc["IoU"],
            "GC_Dice": seg_gc["Dice"],
            "GC_Time": dt_gc,
        })

        # ---- YOLO + SAM ----
        # ---- YOLO + SAM ----
        t0 = time.perf_counter()
        mask_ys = yolo_sam.infer(img_np)
        t1 = time.perf_counter()
        dt_ys = t1 - t0

        mask_ys_bin = mask_ys > 0.5
        seg_ys = compute_segmentation_metrics(mask_ys_bin, gt_mask)

        sums["YOLO+SAM"]["IoU"]  += seg_ys["IoU"]
        sums["YOLO+SAM"]["Dice"] += seg_ys["Dice"]
        sums["YOLO+SAM"]["Time"] += dt_ys

        per_image_entry.update({
            "YS_IoU":  seg_ys["IoU"],
            "YS_Dice": seg_ys["Dice"],
            "YS_Time": dt_ys,
        })


        results_per_image.append(per_image_entry)

        print(f"[{idx+1}/{len(img_paths)}] {basename} done.")

    num_samples = len(results_per_image)
    if num_samples == 0:
        print("No valid image/alpha pairs processed.")
        return

    print("\n==== Summary (averaged over {} images) ====".format(num_samples))
    for m in methods:
        print(f"\n--- {m} ---")
        for mm in mask_metrics:  # ["IoU", "Dice"]
            avg_val = sums[m][mm] / num_samples
            print(f"{mm}: {avg_val:.4f}")
        avg_time = sums[m]["Time"] / num_samples
        print(f"Avg Time per image: {avg_time:.4f} s")


    if output_csv is not None:
        import csv
        fieldnames = sorted(results_per_image[0].keys())
        with open(output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in results_per_image:
                writer.writerow(row)
        print(f"\nPer-image metrics saved to {output_csv}")


# =========================
# CLI
# =========================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate RVM, GrabCut, YOLO+SAM on withoutBG100."
    )
    parser.add_argument("--data-root", type=str, required=True,
                        help="Root folder of withoutbg100 dataset.")
    parser.add_argument("--images-subdir", type=str, default="images",
                        help="Subdir under data-root containing RGB images.")
    parser.add_argument("--alpha-subdir", type=str, default="alpha",
                        help="Subdir under data-root containing alpha mattes.")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device for torch models (cuda or cpu).")
    parser.add_argument("--yolo-weights", type=str, default="yolov8s.pt",
                        help="YOLOv8 weights path, e.g., yolov8s.pt.")
    parser.add_argument("--sam-checkpoint", type=str, default="sam_vit_b.pth",
                        help="Path to SAM checkpoint. If not exists, auto-download.")
    parser.add_argument("--sam-model-type", type=str, default="vit_b",
                        help="SAM model type: vit_b / vit_l / vit_h.")
    parser.add_argument("--target-width", type=int, default=512,
                        help="Resize width for evaluation.")
    parser.add_argument("--target-height", type=int, default=512,
                        help="Resize height for evaluation.")
    parser.add_argument("--output-csv", type=str, default=None,
                        help="Optional CSV file to save per-image metrics.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate_on_dataset(
        data_root=args.data_root,
        images_subdir=args.images_subdir,
        alpha_subdir=args.alpha_subdir,
        device=args.device,
        yolo_weights=args.yolo_weights,
        sam_checkpoint=args.sam_checkpoint,
        sam_model_type=args.sam_model_type,
        target_width=args.target_width,
        target_height=args.target_height,
        output_csv=args.output_csv,
    )
