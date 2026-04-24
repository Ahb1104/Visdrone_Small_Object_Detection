"""
Train ResNet-50 + Faster R-CNN on VisDrone with configurable box predictor head
and training mode.

Three experimental variations:
  Variation 1 -- end_to_end      : ResNet+FPN+RPN+custom head, differential LR
  Variation 2 -- frozen_backbone : RPN+custom head only, ResNet+FPN frozen
  Variation 3 -- backbone_only   : ResNet only, FPN+RPN+standard head frozen

--head standard : FastRCNNPredictor (single FC layer) -- used in backbone_only
--head custom   : ConvRoIHead + CustomDNNBoxPredictor  -- Variations 1 and 2
                  ConvRoIHead replaces TwoMLPHead (box_head) with two conv layers
                  + flatten preserving full 7x7 spatial info for cls and reg.
                  CustomDNNBoxPredictor replaces box_predictor with SE channel
                  attention + residual FC block + separate cls/reg heads.

Designed for Northeastern OOD Explorer cluster.

Usage:
    python train_resnet50_frcnn_dnn.py --head custom   --training_mode end_to_end
    python train_resnet50_frcnn_dnn.py --head custom   --training_mode frozen_backbone
    python train_resnet50_frcnn_dnn.py --head standard --training_mode backbone_only
"""

import argparse
import json
import os
import random
import time
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.ops import box_iou
from PIL import Image, ImageDraw

from torchmetrics.detection import MeanAveragePrecision
from sklearn.metrics import classification_report, confusion_matrix, f1_score


# ============================================================
# Constants
# ============================================================

CLASS_NAMES = [
    "pedestrian", "people", "bicycle", "car", "van",
    "truck", "tricycle", "awning-tricycle", "bus", "motor",
]
N_CLS        = 10
N_CLS_WITH_BG = 11
VALID_CATS   = set(range(1, 11))

DATA_ROOT      = Path("/scratch/<username>/VisDrone/data")
CHECKPOINT_DIR = Path("/scratch/<username>/VisDrone/checkpoints")
PLOT_DIR       = Path.home() / "visdrone" / "plots"

SPLIT_DIRS = {
    "train": "VisDrone2019-DET-train",
    "val":   "VisDrone2019-DET-val",
    "test":  "VisDrone2019-DET-test-dev",
}


# ============================================================
# Custom head building blocks
# ============================================================

class SEChannelAttention(nn.Module):
    """
    Squeeze-and-Excitation channel attention.

    Reweights the 1024-dim feature vector produced by ConvRoIHead to suppress
    background-dominated channels and amplify class-discriminative ones.
    Applied after flatten + FC in the ROI processing pipeline.

    Parameters
    ----------
    dim : int
        Feature dimension (1024).
    reduction : int
        Bottleneck reduction ratio for the squeeze FC.
    """

    def __init__(self, dim, reduction=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(dim, dim // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(dim // reduction, dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        """
        Apply SE channel gating.

        Parameters
        ----------
        x : Tensor
            Shape (N, dim).

        Returns
        -------
        out : Tensor
            Shape (N, dim), channel-reweighted.
        """
        return x * self.fc(x)


class ResidualRefineBlock(nn.Module):
    """
    Residual FC block with batch norm and dropout for feature refinement.

    Parameters
    ----------
    dim : int
        Hidden dimension (1024).
    dropout : float
        Dropout probability.
    """

    def __init__(self, dim, dropout=0.3):
        super().__init__()
        self.fc1  = nn.Linear(dim, dim)
        self.bn1  = nn.BatchNorm1d(dim)
        self.fc2  = nn.Linear(dim, dim)
        self.bn2  = nn.BatchNorm1d(dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.fc1(x)))
        out = self.drop(out)
        out = self.bn2(self.fc2(out))
        return F.relu(out + residual)


# ============================================================
# ConvRoIHead  (replaces TwoMLPHead as model.roi_heads.box_head)
# ============================================================

class ConvRoIHead(nn.Module):
    """
    Convolutional ROI feature extractor replacing the standard TwoMLPHead.

    The standard TwoMLPHead immediately flattens the (N, 256, 7, 7) RoIAlign
    output and passes it through two FC layers, discarding all spatial structure
    before any processing. This module applies two conv layers on the 7x7
    feature map first, then flattens and projects to out_channels=1024. Because
    flattening happens after the conv layers, the full 7x7 spatial layout is
    preserved throughout conv processing and remains available to the downstream
    box predictor for both classification and regression.

    Pipeline:
        (N, 256, 7, 7)  -- RoIAlign output, 256 FPN channels
          -> Conv2d(256, 256, 3x3, pad=1) -> BN2d -> ReLU  [7x7 preserved]
          -> Conv2d(256, 256, 3x3, pad=1) -> BN2d -> ReLU  [7x7 preserved]
          -> Flatten -> (N, 256*7*7 = 12544)
          -> FC(12544, 1024) -> ReLU
        (N, 1024)  -- fed into CustomDNNBoxPredictor

    Maintaining 256 channels throughout matches the FPN output channel count
    and avoids bottlenecking the spatial representations before flatten. The
    FC(12544, 1024) projection is the same dimensionality as the standard
    TwoMLPHead's first linear layer, making the comparison fair.

    Parameters
    ----------
    in_channels : int
        Input channels from RoIAlign (256 for ResNet-50-FPN).
    out_channels : int
        Output feature dimension fed to the box predictor (1024).
    roi_size : int
        Spatial size of the RoIAlign output grid (7 by default).
    """

    def __init__(self, in_channels=256, out_channels=1024, roi_size=7):
        super().__init__()
        self.out_channels = out_channels

        self.conv_layers = nn.Sequential(
            # first conv: process FPN features while preserving 7x7 spatial layout
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # second conv: deepen representations, 7x7 still fully intact
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        # flatten 256*7*7 = 12544 spatial values, all positions retained
        flat_dim = 256 * roi_size * roi_size
        self.fc = nn.Sequential(
            nn.Linear(flat_dim, out_channels),
            nn.ReLU(inplace=True),
        )
        self._init_weights()

    def _init_weights(self):
        """Kaiming normal for conv/fc, ones/zeros for BN."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Extract spatially-aware ROI features via conv + flatten + FC.

        Parameters
        ----------
        x : Tensor
            RoIAlign output, shape (N_rois, in_channels, roi_size, roi_size).

        Returns
        -------
        features : Tensor
            Shape (N_rois, out_channels).
        """
        x = self.conv_layers(x)     # (N, 256, 7, 7) -- spatial intact
        x = x.flatten(start_dim=1)  # (N, 12544) -- full spatial info retained
        return self.fc(x)           # (N, 1024)


# ============================================================
# CustomDNNBoxPredictor  (replaces model.roi_heads.box_predictor)
# ============================================================

class CustomDNNBoxPredictor(nn.Module):
    """
    DNN box predictor operating on spatially-aware 1024-dim ROI features.

    Receives the 1024-dim output of ConvRoIHead (which retains full 7x7
    spatial layout through conv + flatten). Applies SE channel attention to
    reweight class-discriminative channels, a residual FC block for deeper
    feature processing, then separate classification and regression heads.

    Architecture:
        (N, 1024)  -- from ConvRoIHead
          -> SEChannelAttention(1024, reduction=16)
          -> ResidualRefineBlock(1024, dropout)
          -> Dropout(dropout)
          -> cls_score : FC(1024, num_classes)
          -> bbox_pred : FC(1024, num_classes * 4)

    The multi-scale processing role previously handled by the removed
    MultiScaleFeatureRefinement block is now covered by ConvRoIHead's two
    conv layers operating on the 7x7 spatial grid. This predictor focuses
    on channel reweighting and task-specific head specialisation.

    Parameters
    ----------
    in_channels : int
        Input feature dimension from ConvRoIHead (1024).
    num_classes : int
        Number of classes including background (11 for VisDrone).
    dropout : float
        Dropout probability applied before the output heads.
    """

    def __init__(self, in_channels, num_classes, dropout=0.3):
        super().__init__()
        self.attention = SEChannelAttention(in_channels, reduction=16)
        self.residual  = ResidualRefineBlock(in_channels, dropout)
        self.drop      = nn.Dropout(dropout)
        self.cls_score = nn.Linear(in_channels, num_classes)
        self.bbox_pred = nn.Linear(in_channels, num_classes * 4)
        self._init_weights()

    def _init_weights(self):
        """Kaiming normal for linear layers, ones/zeros for BN."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Predict class scores and box deltas.

        Parameters
        ----------
        x : Tensor
            ROI features from ConvRoIHead, shape (N_rois, in_channels).

        Returns
        -------
        cls_logits : Tensor
            Shape (N_rois, num_classes).
        bbox_deltas : Tensor
            Shape (N_rois, num_classes * 4).
        """
        x = self.attention(x)
        x = self.residual(x)
        x = self.drop(x)
        return self.cls_score(x), self.bbox_pred(x)


# ============================================================
# Model Builder
# ============================================================

def build_model(num_classes, head_type="custom", dropout=0.3):
    """
    Build Faster R-CNN with ResNet-50-FPN and configurable ROI head.

    For head_type="custom", both roi_heads sub-modules are replaced:
      box_head      : ConvRoIHead  (conv layers + flatten -> 1024)
      box_predictor : CustomDNNBoxPredictor  (SE + residual + cls/reg)

    For head_type="standard", only box_predictor is replaced with
    FastRCNNPredictor for the VisDrone class count; TwoMLPHead (box_head)
    stays at its COCO-pretrained weights. This is the Variation 3 baseline
    with zero custom components in the model graph.

    Parameters
    ----------
    num_classes : int
        Number of classes including background (11 for VisDrone).
    head_type : str
        "standard" or "custom".
    dropout : float
        Dropout probability for the custom predictor.

    Returns
    -------
    model : nn.Module
        Faster R-CNN with the selected head configuration.
    """
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

    model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)

    if head_type == "standard":
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    elif head_type == "custom":
        # conv_out_dim=1024 matches TwoMLPHead output dim for a fair comparison
        conv_out_dim = 1024
        model.roi_heads.box_head = ConvRoIHead(
            in_channels=256, out_channels=conv_out_dim)
        model.roi_heads.box_predictor = CustomDNNBoxPredictor(
            conv_out_dim, num_classes, dropout)

    else:
        raise ValueError(f"Unknown head_type: {head_type}")

    return model


# ============================================================
# Dataset
# ============================================================

class VisDroneDataset(Dataset):
    """
    VisDrone detection dataset returning resized images and xyxy targets.

    Reads original VisDrone CSV annotation format: comma-separated with
    absolute pixel coordinates x,y,w,h and category at index 5.

    Parameters
    ----------
    root : Path or str
        Path to split directory containing images/ and annotations/.
    max_size : int
        Resize longest side to this value (never upscales).
    o_augment : bool
        Whether to apply random horizontal flip augmentation.
    max_samples : int or None
        Cap on number of images loaded (None = full split).
    """

    def __init__(self, root, max_size=1024, o_augment=False, max_samples=None):
        self.img_dir   = os.path.join(str(root), "images")
        self.ann_dir   = os.path.join(str(root), "annotations")
        self.max_size  = max_size
        self.o_augment = o_augment
        self.img_files = sorted(
            f for f in os.listdir(self.img_dir) if f.endswith(".jpg"))
        if max_samples is not None:
            self.img_files = self.img_files[:max_samples]
        print(f"  Dataset: {len(self.img_files)} images from {root}")

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img = Image.open(
            os.path.join(self.img_dir, self.img_files[idx])).convert("RGB")
        ow, oh = img.size
        scale  = min(self.max_size / ow, self.max_size / oh, 1.0)
        nw, nh = int(ow * scale), int(oh * scale)
        img    = img.resize((nw, nh), Image.BILINEAR)

        ann_path = os.path.join(
            self.ann_dir,
            os.path.splitext(self.img_files[idx])[0] + ".txt")
        boxes, labels = [], []
        if os.path.exists(ann_path):
            with open(ann_path) as f:
                for line in f:
                    p = line.strip().split(",")
                    if len(p) < 8:
                        continue
                    x, y, w, h = float(p[0]), float(p[1]), float(p[2]), float(p[3])
                    cat = int(p[5])
                    if cat not in VALID_CATS or w < 2 or h < 2:
                        continue
                    # 1-indexed labels; 0 = background in Faster R-CNN
                    boxes.append([x * scale, y * scale,
                                  (x + w) * scale, (y + h) * scale])
                    labels.append(cat)

        if len(boxes) == 0:
            boxes_t  = torch.zeros((0, 4), dtype=torch.float32)
            labels_t = torch.zeros((0,), dtype=torch.long)
        else:
            boxes_t  = torch.tensor(boxes, dtype=torch.float32)
            boxes_t[:, 0].clamp_(0, nw)
            boxes_t[:, 1].clamp_(0, nh)
            boxes_t[:, 2].clamp_(0, nw)
            boxes_t[:, 3].clamp_(0, nh)
            labels_t = torch.tensor(labels, dtype=torch.long)
            valid    = ((boxes_t[:, 2] - boxes_t[:, 0] > 2) &
                        (boxes_t[:, 3] - boxes_t[:, 1] > 2))
            boxes_t  = boxes_t[valid]
            labels_t = labels_t[valid]

        img_t = torchvision.transforms.functional.to_tensor(img)

        if self.o_augment and random.random() < 0.5:
            img_t = torch.flip(img_t, [2])
            if boxes_t.shape[0] > 0:
                x1 = nw - boxes_t[:, 2]
                x2 = nw - boxes_t[:, 0]
                boxes_t[:, 0] = x1
                boxes_t[:, 2] = x2

        target = {
            "boxes":    boxes_t,
            "labels":   labels_t,
            "image_id": torch.tensor([idx]),
            "area":     ((boxes_t[:, 2] - boxes_t[:, 0]) *
                         (boxes_t[:, 3] - boxes_t[:, 1])
                         if boxes_t.shape[0] > 0 else torch.zeros(0)),
            "iscrowd":  torch.zeros(boxes_t.shape[0], dtype=torch.long),
        }
        return img_t, target


def collate_fn(batch):
    """Collate images and targets without stacking."""
    return list(zip(*batch))


# ============================================================
# Training
# ============================================================

def train_one_epoch(model, loader, optimizer, device, epoch):
    """
    Train one epoch of Faster R-CNN.

    Parameters
    ----------
    model : nn.Module
    loader : DataLoader
    optimizer : Optimizer
    device : torch.device
    epoch : int

    Returns
    -------
    avg_loss : float
    component_losses : dict
    """
    model.train()
    running_loss   = 0.0
    component_sums = defaultdict(float)
    nb = 0

    for bi, (images, targets) in enumerate(loader):
        images  = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # skip batches where all images have no GT boxes
        valid = [(img, t) for img, t in zip(images, targets)
                 if t["boxes"].shape[0] > 0]
        if len(valid) == 0:
            continue
        images, targets = zip(*valid)
        images, targets = list(images), list(targets)

        loss_dict  = model(images, targets)
        total_loss = sum(v for v in loss_dict.values())

        optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        running_loss += total_loss.item()
        for k, v in loss_dict.items():
            component_sums[k] += v.item()
        nb += 1

        if bi % 200 == 0:
            ls = "  ".join(f"{k}={v.item():.4f}" for k, v in loss_dict.items())
            print(f"  [{epoch}][{bi}/{len(loader)}] {ls}  "
                  f"total={total_loss.item():.4f}")

    avg = max(nb, 1)
    return running_loss / avg, {k: v / avg for k, v in component_sums.items()}


# ============================================================
# Evaluation
# ============================================================

@torch.no_grad()
def evaluate_map(model, loader, device, desc="val"):
    """
    Compute COCO-style detection mAP via torchmetrics.

    Parameters
    ----------
    model : nn.Module
    loader : DataLoader
    device : torch.device
    desc : str

    Returns
    -------
    results : dict
    """
    model.eval()
    metric = MeanAveragePrecision(
        box_format="xyxy", iou_type="bbox", class_metrics=True)

    for bi, (images, targets) in enumerate(loader):
        images      = [img.to(device) for img in images]
        predictions = model(images)

        preds_cpu = [{
            "boxes":  p["boxes"].cpu(),
            "scores": p["scores"].cpu(),
            "labels": p["labels"].cpu(),
        } for p in predictions]
        targets_cpu = [{
            "boxes":  t["boxes"].cpu(),
            "labels": t["labels"].cpu(),
        } for t in targets]

        metric.update(preds_cpu, targets_cpu)
        if bi % 50 == 0:
            print(f"    [{desc}] {bi}/{len(loader)}")

    results = metric.compute()
    return {k: v.item() if isinstance(v, torch.Tensor) and v.numel() == 1 else v
            for k, v in results.items()}


@torch.no_grad()
def evaluate_classification(model, loader, device, iou_thr=0.5, score_thr=0.3):
    """
    Evaluate classification accuracy on IoU-matched detections.

    Parameters
    ----------
    model : nn.Module
    loader : DataLoader
    device : torch.device
    iou_thr : float
        IoU threshold for matching a prediction to a GT box.
    score_thr : float
        Minimum confidence score to consider a prediction.

    Returns
    -------
    gt_labels : np.ndarray
    pred_labels : np.ndarray
    n_gt : int
    n_pred : int
    n_matched : int
    """
    model.eval()
    all_gt, all_pred = [], []
    n_gt = n_pred = n_matched = 0

    for images, targets in loader:
        images      = [img.to(device) for img in images]
        predictions = model(images)

        for pred, tgt in zip(predictions, targets):
            gb = tgt["boxes"].cpu()
            gl = tgt["labels"].cpu()
            pb = pred["boxes"].cpu()
            ps = pred["scores"].cpu()
            pl = pred["labels"].cpu()

            n_gt += gb.shape[0]
            keep  = ps >= score_thr
            pb, ps, pl = pb[keep], ps[keep], pl[keep]
            n_pred += pb.shape[0]

            if gb.shape[0] == 0 or pb.shape[0] == 0:
                continue

            ious       = box_iou(pb, gb)
            matched_gt = set()
            for pi in ps.argsort(descending=True):
                bv, bi = ious[pi].max(0)
                if bv.item() >= iou_thr and bi.item() not in matched_gt:
                    matched_gt.add(bi.item())
                    all_gt.append(gl[bi].item())
                    all_pred.append(pl[pi].item())
                    n_matched += 1

    return np.array(all_gt), np.array(all_pred), n_gt, n_pred, n_matched


# ============================================================
# Plotting
# ============================================================

def save_fig(fig, path):
    """Save figure and close."""
    fig.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {path}")


def plot_training_curves(history, n_epochs, plot_dir, head_label=""):
    """
    Plot total loss, loss components, mAP progression, and LR schedule.

    Parameters
    ----------
    history : dict
    n_epochs : int
    plot_dir : Path
    head_label : str
    """
    eps = range(1, n_epochs + 1)
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    axes[0, 0].plot(eps, history["train_total"], "b-o", ms=3)
    axes[0, 0].set_title("Total Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].grid(True, alpha=0.3)

    colors = {"loss_classifier": "blue", "loss_box_reg": "red",
              "loss_objectness": "green", "loss_rpn_box_reg": "orange"}
    nice   = {"loss_classifier": "ROI Cls", "loss_box_reg": "ROI Box",
              "loss_objectness": "RPN Obj", "loss_rpn_box_reg": "RPN Box"}
    for k, vals in history.items():
        if k.startswith("comp_"):
            name = k.replace("comp_", "")
            axes[0, 1].plot(eps, vals, "-o", ms=3,
                            color=colors.get(name, "gray"),
                            label=nice.get(name, name))
    axes[0, 1].set_title("Loss Components")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(eps, history["val_map50"],   "g-o", ms=3, lw=2, label="mAP@0.5")
    axes[1, 0].plot(eps, history["val_map5095"], "r-s", ms=3, lw=2, label="mAP@0.5:0.95")
    axes[1, 0].set_title("Detection mAP (Val)")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(eps, history["lr"], "m-o", ms=3)
    axes[1, 1].set_title("Learning Rate")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].grid(True, alpha=0.3)

    title = (f"ResNet-50 + Faster R-CNN + {head_label}"
             if head_label else "ResNet-50 + Faster R-CNN")
    fig.suptitle(title, fontsize=14, y=1.01)
    fig.tight_layout()
    save_fig(fig, plot_dir / "training_curves.png")


def plot_map_bar(final_results, plot_dir, head_label=""):
    """
    Plot COCO detection mAP bar chart, val vs test.

    Parameters
    ----------
    final_results : dict
    plot_dir : Path
    head_label : str
    """
    mk = ["map", "map_50", "map_75", "map_small", "map_medium", "map_large", "mar_100"]
    ml = ["mAP\n@0.5:0.95", "mAP\n@0.5", "mAP\n@0.75",
          "mAP\nsmall", "mAP\nmedium", "mAP\nlarge", "mAR\n@100"]

    def _g(r, k):
        v = r.get(k, 0)
        return max(v.item() if isinstance(v, torch.Tensor) and v.numel() == 1 else v, 0)

    vv = [_g(final_results["val"],  k) for k in mk]
    tv = [_g(final_results["test"], k) for k in mk]
    x  = np.arange(len(mk))
    w  = 0.35

    fig, ax = plt.subplots(figsize=(12, 5))
    bv = ax.bar(x - w / 2, vv, w, label="Val",  color="steelblue")
    bt = ax.bar(x + w / 2, tv, w, label="Test", color="coral")
    ax.set_xticks(x)
    ax.set_xticklabels(ml)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    title = (f"Detection mAP (COCO Protocol) - {head_label}"
             if head_label else "Detection mAP (COCO Protocol)")
    ax.set_title(title)
    for bars in [bv, bt]:
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                        f"{h:.3f}", ha="center", fontsize=8)
    fig.tight_layout()
    save_fig(fig, plot_dir / "detection_map.png")


def plot_per_class_ap(final_results, plot_dir):
    """
    Plot per-class AP bar chart for val and test splits.

    Parameters
    ----------
    final_results : dict
    plot_dir : Path
    """
    for sn in ["val", "test"]:
        r   = final_results[sn]
        mpc = r.get("map_per_class", None)
        if mpc is None or (isinstance(mpc, torch.Tensor) and mpc.numel() == 1):
            continue
        ap = mpc.cpu().numpy() if isinstance(mpc, torch.Tensor) else np.array(mpc)
        ci = r.get("classes", torch.arange(1, N_CLS_WITH_BG))
        ci = (ci.cpu().numpy() if isinstance(ci, torch.Tensor)
              else np.arange(1, N_CLS_WITH_BG))

        fig, ax = plt.subplots(figsize=(10, 6))
        y    = np.arange(len(ap))
        bars = ax.barh(y, ap,
                       color=plt.cm.viridis(np.linspace(0.2, 0.8, len(ap))))
        lbl  = [CLASS_NAMES[int(c) - 1] if 1 <= int(c) <= N_CLS else f"c{c}"
                for c in ci]
        ax.set_yticks(y)
        ax.set_yticklabels(lbl)
        ax.set_xlabel("AP @0.5:0.95")
        ax.set_title(f"Per-Class AP - {sn}")
        ax.set_xlim(0, 1)
        ax.grid(True, axis="x", alpha=0.3)
        for b, v in zip(bars, ap):
            ax.text(b.get_width() + 0.01, b.get_y() + b.get_height() / 2,
                    f"{v:.3f}", va="center", fontsize=9)
        m = np.mean(ap[ap >= 0])
        ax.axvline(m, color="red", ls="--", lw=1.5, label=f"Mean:{m:.3f}")
        ax.legend(loc="lower right")
        fig.tight_layout()
        save_fig(fig, plot_dir / f"per_class_ap_{sn}.png")


def plot_confusion_matrices(cls_data, plot_dir):
    """
    Plot raw and normalised confusion matrices on IoU-matched detections.

    Parameters
    ----------
    cls_data : dict
    plot_dir : Path
    """
    for sn in ["val", "test"]:
        cd = cls_data[sn]
        gt, pred = cd["gt"], cd["pred"]
        if len(gt) == 0:
            continue
        cls_ids = list(range(1, N_CLS_WITH_BG))
        cm      = confusion_matrix(gt, pred, labels=cls_ids)
        cm_n    = cm.astype(np.float32) / cm.sum(1, keepdims=True).clip(1)

        fig, axes = plt.subplots(1, 2, figsize=(22, 8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                    ax=axes[0])
        axes[0].set_title(f"Confusion Matrix (Counts) - {sn}")
        axes[0].set_xlabel("Predicted")
        axes[0].set_ylabel("True")
        axes[0].tick_params(axis="x", rotation=45)

        sns.heatmap(cm_n, annot=True, fmt=".2f", cmap="Blues",
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                    ax=axes[1], vmin=0, vmax=1)
        axes[1].set_title(f"Confusion Matrix (Normalized) - {sn}")
        axes[1].set_xlabel("Predicted")
        axes[1].set_ylabel("True")
        axes[1].tick_params(axis="x", rotation=45)
        fig.tight_layout()
        save_fig(fig, plot_dir / f"confusion_matrix_{sn}.png")


def plot_cls_accuracy(cls_data, plot_dir):
    """
    Plot per-class classification accuracy on IoU-matched detections.

    Parameters
    ----------
    cls_data : dict
    plot_dir : Path
    """
    for sn in ["val", "test"]:
        cd = cls_data[sn]
        gt, pred = cd["gt"], cd["pred"]
        if len(gt) == 0:
            continue
        accs, ns = [], []
        for c in range(1, N_CLS_WITH_BG):
            m = gt == c
            n = m.sum()
            accs.append((pred[m] == c).sum() / max(n, 1))
            ns.append(n)

        fig, ax = plt.subplots(figsize=(10, 6))
        y    = np.arange(N_CLS)
        bars = ax.barh(y, accs,
                       color=plt.cm.viridis(np.linspace(0.2, 0.8, N_CLS)))
        ax.set_yticks(y)
        ax.set_yticklabels(
            [f"{CLASS_NAMES[c]} (n={ns[c]})" for c in range(N_CLS)])
        ax.set_xlabel("Accuracy")
        ax.set_title(f"Per-Class Cls Accuracy (Matched) - {sn}")
        ax.set_xlim(0, 1)
        ax.grid(True, axis="x", alpha=0.3)
        for b, a in zip(bars, accs):
            ax.text(b.get_width() + 0.01, b.get_y() + b.get_height() / 2,
                    f"{a:.3f}", va="center", fontsize=9)
        ov = (gt == pred).mean()
        ax.axvline(ov, color="red", ls="--", lw=1.5,
                   label=f"Overall:{ov:.3f}")
        ax.legend(loc="lower right")
        fig.tight_layout()
        save_fig(fig, plot_dir / f"cls_acc_matched_{sn}.png")


@torch.no_grad()
def collect_all_predictions(model, loader, device, score_thr=0.01):
    """
    Collect all predictions and GT across the dataset for curve plotting.

    Parameters
    ----------
    model : nn.Module
    loader : DataLoader
    device : torch.device
    score_thr : float
        Low threshold retains the full confidence range for curve plots.

    Returns
    -------
    all_preds : list of dict
    all_gts : list of dict
    """
    model.eval()
    all_preds, all_gts = [], []
    for images, targets in loader:
        images      = [img.to(device) for img in images]
        predictions = model(images)
        for pred, tgt in zip(predictions, targets):
            keep = pred["scores"].cpu() >= score_thr
            all_preds.append({
                "boxes":  pred["boxes"].cpu()[keep],
                "scores": pred["scores"].cpu()[keep],
                "labels": pred["labels"].cpu()[keep],
            })
            all_gts.append({
                "boxes":  tgt["boxes"].cpu(),
                "labels": tgt["labels"].cpu(),
            })
    return all_preds, all_gts


def compute_per_class_curves(all_preds, all_gts, iou_thr=0.5):
    """
    Compute per-class precision, recall, F1 as functions of confidence threshold.

    Parameters
    ----------
    all_preds : list of dict
    all_gts : list of dict
    iou_thr : float

    Returns
    -------
    curves : dict
        Keys are class indices (1-indexed). Each value contains:
        scores, precision, recall, f1, n_gt, ap.
    """
    curves = {}
    for cls in range(1, N_CLS_WITH_BG):
        all_scores, all_tp, n_gt = [], [], 0

        for pred, gt in zip(all_preds, all_gts):
            gt_mask  = gt["labels"] == cls
            gt_boxes = gt["boxes"][gt_mask]
            n_gt    += gt_boxes.shape[0]

            pred_mask = pred["labels"] == cls
            p_boxes   = pred["boxes"][pred_mask]
            p_scores  = pred["scores"][pred_mask]
            if p_boxes.shape[0] == 0:
                continue

            order    = p_scores.argsort(descending=True)
            p_boxes  = p_boxes[order]
            p_scores = p_scores[order]

            matched = set()
            tp = np.zeros(len(p_scores))
            if gt_boxes.shape[0] > 0:
                iou_mat = box_iou(p_boxes, gt_boxes)
                for i in range(len(p_scores)):
                    best_iou, best_j = iou_mat[i].max(0)
                    if best_iou.item() >= iou_thr and best_j.item() not in matched:
                        tp[i] = 1
                        matched.add(best_j.item())

            all_scores.extend(p_scores.numpy().tolist())
            all_tp.extend(tp.tolist())

        if n_gt == 0:
            curves[cls] = {"scores": [], "precision": [], "recall": [],
                           "f1": [], "n_gt": 0, "ap": 0.0}
            continue

        order         = np.argsort(all_scores)[::-1]
        sorted_scores = np.array(all_scores)[order]
        tp_cum        = np.cumsum(np.array(all_tp)[order])
        fp_cum        = np.cumsum(1 - np.array(all_tp)[order])

        precision = tp_cum / (tp_cum + fp_cum + 1e-8)
        recall    = tp_cum / (n_gt + 1e-8)
        f1        = 2 * precision * recall / (precision + recall + 1e-8)

        # AP via all-point interpolation
        mrec = np.concatenate(([0.0], recall, [1.0]))
        mpre = np.concatenate(([1.0], precision, [0.0]))
        for i in range(len(mpre) - 1, 0, -1):
            mpre[i - 1] = max(mpre[i - 1], mpre[i])
        idx = np.where(mrec[1:] != mrec[:-1])[0]
        ap  = np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])

        curves[cls] = {
            "scores":    sorted_scores.tolist(),
            "precision": precision.tolist(),
            "recall":    recall.tolist(),
            "f1":        f1.tolist(),
            "n_gt":      n_gt,
            "ap":        float(ap),
        }

    return curves


def plot_pr_curves_per_class(curves, plot_dir, split_name):
    """Plot Precision-Recall curve per class."""
    fig, ax = plt.subplots(figsize=(10, 7))
    palette = plt.cm.tab10(np.linspace(0, 1, N_CLS))
    for cls in range(1, N_CLS_WITH_BG):
        c = curves.get(cls, {})
        if len(c.get("recall", [])) == 0:
            continue
        ax.plot(c["recall"], c["precision"],
                color=palette[cls - 1], linewidth=1.5,
                label=f"{CLASS_NAMES[cls-1]} (AP={c['ap']:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall Curve per Class - {split_name}")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_fig(fig, plot_dir / f"pr_curve_per_class_{split_name}.png")


def plot_f1_confidence_per_class(curves, plot_dir, split_name):
    """Plot F1 score vs confidence threshold per class."""
    fig, ax = plt.subplots(figsize=(10, 7))
    palette = plt.cm.tab10(np.linspace(0, 1, N_CLS))
    for cls in range(1, N_CLS_WITH_BG):
        c = curves.get(cls, {})
        if len(c.get("scores", [])) == 0:
            continue
        best_idx  = np.argmax(c["f1"])
        ax.plot(c["scores"], c["f1"],
                color=palette[cls - 1], linewidth=1.5,
                label=(f"{CLASS_NAMES[cls-1]} "
                       f"(best={c['f1'][best_idx]:.3f}"
                       f"@{c['scores'][best_idx]:.2f})"))
    ax.set_xlabel("Confidence Threshold")
    ax.set_ylabel("F1 Score")
    ax.set_title(f"F1-Confidence Curve per Class - {split_name}")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend(loc="upper right", fontsize=7)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_fig(fig, plot_dir / f"f1_confidence_per_class_{split_name}.png")


def plot_precision_confidence_per_class(curves, plot_dir, split_name):
    """Plot Precision vs confidence threshold per class."""
    fig, ax = plt.subplots(figsize=(10, 7))
    palette = plt.cm.tab10(np.linspace(0, 1, N_CLS))
    for cls in range(1, N_CLS_WITH_BG):
        c = curves.get(cls, {})
        if len(c.get("scores", [])) == 0:
            continue
        ax.plot(c["scores"], c["precision"],
                color=palette[cls - 1], linewidth=1.5,
                label=CLASS_NAMES[cls - 1])
    ax.set_xlabel("Confidence Threshold")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Confidence Curve per Class - {split_name}")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_fig(fig, plot_dir / f"precision_confidence_per_class_{split_name}.png")


def plot_recall_confidence_per_class(curves, plot_dir, split_name):
    """Plot Recall vs confidence threshold per class."""
    fig, ax = plt.subplots(figsize=(10, 7))
    palette = plt.cm.tab10(np.linspace(0, 1, N_CLS))
    for cls in range(1, N_CLS_WITH_BG):
        c = curves.get(cls, {})
        if len(c.get("scores", [])) == 0:
            continue
        ax.plot(c["scores"], c["recall"],
                color=palette[cls - 1], linewidth=1.5,
                label=CLASS_NAMES[cls - 1])
    ax.set_xlabel("Confidence Threshold")
    ax.set_ylabel("Recall")
    ax.set_title(f"Recall-Confidence Curve per Class - {split_name}")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_fig(fig, plot_dir / f"recall_confidence_per_class_{split_name}.png")


def plot_ap_aucpr_bar(curves, plot_dir, split_name):
    """
    Plot per-class AP (AUC of PR curve) as a bar chart with mean line.

    Parameters
    ----------
    curves : dict
    plot_dir : Path
    split_name : str

    Returns
    -------
    mean_ap : float
    aps : list of float
    """
    aps   = [curves.get(cls, {}).get("ap", 0.0)
             for cls in range(1, N_CLS_WITH_BG)]
    names = [CLASS_NAMES[cls - 1] for cls in range(1, N_CLS_WITH_BG)]

    fig, ax    = plt.subplots(figsize=(10, 6))
    y          = np.arange(len(aps))
    colors_bar = plt.cm.viridis(np.linspace(0.2, 0.8, len(aps)))
    bars       = ax.barh(y, aps, color=colors_bar)
    ax.set_yticks(y)
    ax.set_yticklabels(
        [f"{n} (n={curves.get(c+1,{}).get('n_gt',0)})"
         for c, n in enumerate(names)])
    ax.set_xlabel("AP (AUC of PR Curve)")
    ax.set_title(f"Per-Class AP (AUC-PR) @IoU=0.5 - {split_name}")
    ax.set_xlim(0, 1)
    ax.grid(True, axis="x", alpha=0.3)
    for bar, v in zip(bars, aps):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{v:.3f}", va="center", fontsize=9)
    mean_ap = np.mean(aps)
    ax.axvline(mean_ap, color="red", ls="--", lw=1.5,
               label=f"Mean AP: {mean_ap:.3f}")
    ax.legend(loc="lower right")
    fig.tight_layout()
    save_fig(fig, plot_dir / f"ap_aucpr_per_class_{split_name}.png")
    return mean_ap, aps


def plot_sample_detections(model, dataset, device, plot_dir, n=8, thr=0.4):
    """
    Draw GT and predicted boxes side by side for n sample images.

    Each row shows one image: left column = GT boxes with class labels,
    right column = predicted boxes with class label and confidence score.
    Both columns share the same per-class color palette for direct comparison.

    Parameters
    ----------
    model : nn.Module
    dataset : Dataset
    device : torch.device
    plot_dir : Path
    n : int
        Number of sample images to visualise.
    thr : float
        Minimum prediction confidence to display.
    """
    model.eval()
    idxs    = random.sample(range(len(dataset)), min(n, len(dataset)))
    palette = [
        (255, 0,   0),   (0, 200, 0),   (0,   0, 255), (255, 200, 0),
        (255, 0, 255),   (0, 220, 220), (180,  0,   0), (0,  140,  0),
        (0,   0, 180),   (180, 180, 0),
    ]

    # squeeze=False guarantees axes shape is always (n, 2) even when n=1
    fig, axes = plt.subplots(n, 2, figsize=(20, n * 5), squeeze=False)

    with torch.no_grad():
        for row, idx in enumerate(idxs):
            img_t, tgt = dataset[idx]
            pred       = model([img_t.to(device)])[0]
            img_np     = (img_t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

            # --- left: ground truth ---
            img_gt  = Image.fromarray(img_np.copy())
            draw_gt = ImageDraw.Draw(img_gt)
            for b, lbl in zip(tgt["boxes"].tolist(), tgt["labels"].tolist()):
                c   = lbl - 1
                col = palette[c % 10] if 0 <= c < 10 else (255, 255, 255)
                draw_gt.rectangle(b, outline=col, width=2)
                name = CLASS_NAMES[c] if 0 <= c < N_CLS else f"c{lbl}"
                draw_gt.text((b[0], max(b[1] - 12, 0)), name, fill=col)
            axes[row, 0].imshow(np.array(img_gt))
            axes[row, 0].axis("off")
            axes[row, 0].set_title(
                f"GT  (n={tgt['boxes'].shape[0]})", fontsize=9)

            # --- right: predictions ---
            img_pred  = Image.fromarray(img_np.copy())
            draw_pred = ImageDraw.Draw(img_pred)
            keep      = pred["scores"].cpu() >= thr
            n_kept    = keep.sum().item()
            for b, lbl, s in zip(pred["boxes"].cpu()[keep].tolist(),
                                  pred["labels"].cpu()[keep].tolist(),
                                  pred["scores"].cpu()[keep].tolist()):
                c   = lbl - 1
                col = palette[c % 10] if 0 <= c < 10 else (255, 255, 255)
                draw_pred.rectangle(b, outline=col, width=2)
                name = CLASS_NAMES[c] if 0 <= c < N_CLS else f"c{lbl}"
                draw_pred.text((b[0], max(b[1] - 12, 0)),
                               f"{name} {s:.2f}", fill=col)
            axes[row, 1].imshow(np.array(img_pred))
            axes[row, 1].axis("off")
            axes[row, 1].set_title(
                f"Pred  (n={n_kept}, thr={thr})", fontsize=9)

    fig.suptitle("Sample Detections -- GT (left) vs Predicted (right)",
                 fontsize=14)
    fig.tight_layout()
    save_fig(fig, plot_dir / "sample_detections.png")


# ============================================================
# Training-mode helpers
# ============================================================

def freeze_params(model, *, freeze_backbone=False,
                  freeze_fpn_rpn=False, freeze_head=False):
    """
    Selectively freeze parameter groups.

    Torchvision packages ResNet-50 and FPN together under model.backbone
    (BackboneWithFPN). Parameter name conventions:
      "backbone"  -- ResNet-50 body + FPN lateral/output layers
      "rpn"       -- Region Proposal Network
      "roi_heads" -- box_head (ConvRoIHead or TwoMLPHead) + box_predictor

    Parameters
    ----------
    model : nn.Module
    freeze_backbone : bool
        Freeze ResNet-50 + FPN (model.backbone).
    freeze_fpn_rpn : bool
        Freeze FPN and RPN independently of backbone body. Used in
        backbone_only mode where backbone trains but FPN+RPN are frozen.
    freeze_head : bool
        Freeze roi_heads entirely (box_head + box_predictor).
    """
    for name, param in model.named_parameters():
        is_backbone = "backbone" in name
        is_fpn_rpn  = ("rpn" in name) or ("fpn" in name)
        is_head     = "roi_heads" in name
        if freeze_backbone and is_backbone:
            param.requires_grad_(False)
        if freeze_fpn_rpn and is_fpn_rpn:
            param.requires_grad_(False)
        if freeze_head and is_head:
            param.requires_grad_(False)


def _build_optimizer(model, mode, lr_backbone, lr_head):
    """
    Return an AdamW optimizer with parameter groups matching the training mode.

    end_to_end / end_to_end_then_finetune (phase 1)
        Two groups: backbone at lr_backbone, everything else at lr_head.
    frozen_backbone
        Single group: all requires_grad params at lr_head.
    backbone_only
        Single group: backbone params only at lr_backbone.

    Parameters
    ----------
    model : nn.Module
    mode : str
    lr_backbone : float
    lr_head : float

    Returns
    -------
    optimizer : torch.optim.AdamW
    """
    if mode == "frozen_backbone":
        trainable = [p for p in model.parameters() if p.requires_grad]
        return torch.optim.AdamW(trainable, lr=lr_head, weight_decay=1e-4)

    elif mode == "backbone_only":
        bb = [p for n, p in model.named_parameters()
              if "backbone" in n and p.requires_grad]
        return torch.optim.AdamW(bb, lr=lr_backbone, weight_decay=1e-4)

    else:  # end_to_end, end_to_end_then_finetune
        bb = [p for n, p in model.named_parameters()
              if "backbone" in n and p.requires_grad]
        hd = [p for n, p in model.named_parameters()
              if "backbone" not in n and p.requires_grad]
        return torch.optim.AdamW(
            [{"params": bb, "lr": lr_backbone},
             {"params": hd, "lr": lr_head}],
            weight_decay=1e-4,
        )


def run_training_loop(
    model, train_loader, val_loader, optimizer, scheduler,
    start_epoch, n_epochs, tag, device, history, checkpoint_dir,
    phase_label="",
):
    """
    Execute the training loop for n_epochs epochs, updating history in place.

    Parameters
    ----------
    model : nn.Module
    train_loader : DataLoader
    val_loader : DataLoader
    optimizer : torch.optim.Optimizer
    scheduler : LR scheduler
    start_epoch : int
        0-based epoch index to begin at (handles resume and phase-2 offset).
    n_epochs : int
        Number of epochs to run in this phase.
    tag : str
        Run identifier used in checkpoint filenames.
    device : torch.device
    history : defaultdict(list)
        Mutable accumulator passed in and returned updated.
    checkpoint_dir : Path
    phase_label : str
        Label printed in epoch headers.

    Returns
    -------
    best_map50 : float
    best_epoch : int
        1-indexed absolute epoch number of the best checkpoint.
    """
    best_map50 = 0.0
    best_epoch = 0
    n_groups   = len(optimizer.param_groups)

    for epoch in range(start_epoch, start_epoch + n_epochs):
        t0 = time.time()
        print(f"\n{'='*60}")
        ph = f"[{phase_label}] " if phase_label else ""
        if n_groups >= 2:
            print(f"{ph}Epoch {epoch+1}  "
                  f"lr_bb={optimizer.param_groups[0]['lr']:.2e}  "
                  f"lr_head={optimizer.param_groups[1]['lr']:.2e}")
        else:
            print(f"{ph}Epoch {epoch+1}  "
                  f"lr={optimizer.param_groups[0]['lr']:.2e}")
        print(f"{'='*60}")

        avg_loss, comp = train_one_epoch(
            model, train_loader, optimizer, device, epoch + 1)
        scheduler.step()

        history["train_total"].append(avg_loss)
        for k, v in comp.items():
            history[f"comp_{k}"].append(v)
        history["lr"].append(optimizer.param_groups[-1]["lr"])

        print(f"  evaluating val...")
        val_res = evaluate_map(model, val_loader, device, "val")
        m50     = val_res.get("map_50", 0.0)
        m5095   = val_res.get("map", 0.0)
        history["val_map50"].append(m50)
        history["val_map5095"].append(m5095)

        elapsed  = time.time() - t0
        comp_str = "  ".join(f"{k}={v:.4f}" for k, v in comp.items())
        print(f"  loss={avg_loss:.4f}  ({comp_str})")
        print(f"  mAP@0.5={m50:.4f}  mAP@0.5:0.95={m5095:.4f}")
        print(f"  mAP@0.75={val_res.get('map_75', 0):.4f}  "
              f"mAR@100={val_res.get('mar_100', 0):.4f}")
        print(f"  time={elapsed:.0f}s")

        ckpt_data = {
            "epoch":                epoch,
            "model_state_dict":     model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "map_50":               m50,
        }
        torch.save(ckpt_data,
                   checkpoint_dir / f"frcnn_{tag}_ep{epoch+1:03d}.pt")
        if m50 > best_map50:
            best_map50 = m50
            best_epoch = epoch + 1
            torch.save(ckpt_data, checkpoint_dir / f"frcnn_{tag}_best.pt")
            print(f"  >> new best mAP@0.5={best_map50:.4f}")

    return best_map50, best_epoch


# ============================================================
# Main
# ============================================================

def parse_args():
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description="ResNet-50 + Faster R-CNN on VisDrone -- three experimental variations")
    p.add_argument("--head", type=str, default="custom",
                   choices=["standard", "custom"])
    p.add_argument("--training_mode", type=str, default="end_to_end",
                   choices=[
                       "end_to_end",               # Variation 1
                       "frozen_backbone",           # Variation 2
                       "backbone_only",             # Variation 3
                       "end_to_end_then_finetune",  # optional two-phase
                   ])
    p.add_argument("--epochs",            type=int,   default=30)
    p.add_argument("--finetune_epochs",   type=int,   default=20)
    p.add_argument("--batch_size",        type=int,   default=4)
    p.add_argument("--img_max_size",      type=int,   default=1024)
    p.add_argument("--lr_backbone",       type=float, default=5e-5)
    p.add_argument("--lr_head",           type=float, default=5e-4)
    p.add_argument("--dropout",           type=float, default=0.3)
    p.add_argument("--num_workers",       type=int,   default=4)
    p.add_argument("--max_train_samples", type=int,   default=None)
    p.add_argument("--resume",            type=str,   default=None)
    return p.parse_args()


def main():
    """
    Build, train, evaluate, and produce all plots/metrics.

    Experimental variations
    -----------------------
    end_to_end  (Variation 1)
        Full fine-tuning with differential LRs. ResNet-50+FPN backbone at
        lr_backbone; RPN + ConvRoIHead + CustomDNNBoxPredictor at lr_head.

    frozen_backbone  (Variation 2)
        ResNet-50+FPN frozen at COCO weights. RPN + ConvRoIHead +
        CustomDNNBoxPredictor train at lr_head. Isolates custom head
        contribution independent of backbone domain adaptation.

    backbone_only  (Variation 3)
        FPN + RPN + standard FastRCNNPredictor frozen at COCO weights. Only
        ResNet-50 backbone fine-tunes at lr_backbone. No custom components
        present. Isolates backbone adaptation contribution.

    end_to_end_then_finetune  (optional)
        Phase 1: end_to_end for --epochs.
        Phase 2: backbone frozen, head fine-tunes for --finetune_epochs at
        0.1 * lr_head with a fresh cosine schedule.
    """
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}  training_mode={args.training_mode}  head={args.head}")
    if torch.cuda.is_available():
        print(f"gpu={torch.cuda.get_device_name(0)}")

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- data ----
    print("Loading datasets...")
    train_ds = VisDroneDataset(
        DATA_ROOT / SPLIT_DIRS["train"],
        args.img_max_size, True, args.max_train_samples)
    val_ds   = VisDroneDataset(DATA_ROOT / SPLIT_DIRS["val"],  args.img_max_size)
    test_ds  = VisDroneDataset(DATA_ROOT / SPLIT_DIRS["test"], args.img_max_size)

    train_loader = DataLoader(
        train_ds, args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True)
    val_loader   = DataLoader(
        val_ds, args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True)
    test_loader  = DataLoader(
        test_ds, args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True)

    # ---- model ----
    model      = build_model(
        N_CLS_WITH_BG, head_type=args.head, dropout=args.dropout).to(device)
    head_label = ("Standard FastRCNNPredictor" if args.head == "standard"
                  else "Custom Conv+DNN (ConvRoIHead+SE+Residual)")
    tag        = f"{args.head}_{args.training_mode}"

    # ---- freeze params per training mode ----
    if args.training_mode == "frozen_backbone":
        freeze_params(model, freeze_backbone=True)
        trainable_desc = "RPN + ConvRoIHead + CustomDNNBoxPredictor  [backbone frozen]"

    elif args.training_mode == "end_to_end":
        trainable_desc = ("backbone (lr_bb) + RPN + "
                          "ConvRoIHead + CustomDNNBoxPredictor (lr_head)")

    elif args.training_mode == "backbone_only":
        freeze_params(model, freeze_fpn_rpn=True, freeze_head=True)
        trainable_desc = "ResNet-50 backbone only  [FPN + RPN + head frozen]"

    elif args.training_mode == "end_to_end_then_finetune":
        trainable_desc = "Phase-1: end-to-end  |  Phase-2: RPN + head only"

    else:
        raise ValueError(f"Unknown training_mode: {args.training_mode}")

    n_p     = sum(p.numel() for p in model.parameters()) / 1e6
    n_head  = sum(p.numel() for p in model.roi_heads.parameters()) / 1e6
    n_train = sum(p.numel() for p in model.parameters()
                  if p.requires_grad) / 1e6
    print(f"\nModel : Faster R-CNN + ResNet-50-FPN + {head_label}")
    print(f"  Total params     : {n_p:.2f}M")
    print(f"  ROI head params  : {n_head:.4f}M  (box_head + box_predictor)")
    print(f"  Trainable params : {n_train:.3f}M  ({trainable_desc})")

    # ---- optimizer / scheduler ----
    optimizer = _build_optimizer(
        model, args.training_mode, args.lr_backbone, args.lr_head)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6)

    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        print(f"  resumed from epoch {start_epoch}")

    history = defaultdict(list)

    # ================================================================
    # Phase 1
    # ================================================================
    phase1_label = ("Phase-1" if args.training_mode == "end_to_end_then_finetune"
                    else "")
    best_map50, best_epoch = run_training_loop(
        model, train_loader, val_loader,
        optimizer, scheduler,
        start_epoch, args.epochs,
        tag, device, history, CHECKPOINT_DIR,
        phase_label=phase1_label,
    )
    print(f"\nPhase-1 complete.  Best mAP@0.5={best_map50:.4f} at epoch {best_epoch}")

    # ================================================================
    # Phase 2  (end_to_end_then_finetune only)
    # ================================================================
    if args.training_mode == "end_to_end_then_finetune":
        print(f"\n{'='*60}")
        print(f"Phase-2: head-only fine-tuning for {args.finetune_epochs} epochs")
        print(f"  Starting from best phase-1 checkpoint (epoch {best_epoch})")
        print(f"{'='*60}")

        ckpt = torch.load(CHECKPOINT_DIR / f"frcnn_{tag}_best.pt",
                          map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])

        freeze_params(model, freeze_backbone=True)
        n_train_p2 = sum(p.numel() for p in model.parameters()
                         if p.requires_grad) / 1e6
        print(f"  Trainable params phase-2 : {n_train_p2:.3f}M  (RPN + head)")

        p2_lr        = args.lr_head * 0.1
        p2_params    = [p for p in model.parameters() if p.requires_grad]
        optimizer_p2 = torch.optim.AdamW(p2_params, lr=p2_lr, weight_decay=1e-4)
        scheduler_p2 = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer_p2, T_max=args.finetune_epochs, eta_min=1e-7)

        history2 = defaultdict(list)
        best_map50_p2, best_epoch_p2 = run_training_loop(
            model, train_loader, val_loader,
            optimizer_p2, scheduler_p2,
            start_epoch=args.epochs,
            n_epochs=args.finetune_epochs,
            tag=tag, device=device,
            history=history2,
            checkpoint_dir=CHECKPOINT_DIR,
            phase_label="Phase-2",
        )
        for k, v in history2.items():
            history[k].extend(v)

        if best_map50_p2 > best_map50:
            best_map50 = best_map50_p2
            best_epoch = best_epoch_p2
        total_epochs_trained = args.epochs + args.finetune_epochs
        print(f"\nPhase-2 complete.  "
              f"Overall best mAP@0.5={best_map50:.4f} at epoch {best_epoch}")
    else:
        total_epochs_trained = args.epochs

    # ================================================================
    # Final evaluation from best checkpoint
    # ================================================================
    ckpt = torch.load(CHECKPOINT_DIR / f"frcnn_{tag}_best.pt",
                      map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded best checkpoint (epoch {best_epoch})")

    final_results, cls_data = {}, {}

    for sn, loader in [("val", val_loader), ("test", test_loader)]:
        print(f"\nFinal detection mAP on {sn}...")
        res = evaluate_map(model, loader, device, sn)
        final_results[sn] = res
        print(f"  mAP@0.5={res.get('map_50',0):.4f}  "
              f"mAP@0.5:0.95={res.get('map',0):.4f}")
        print(f"  mAP@0.75={res.get('map_75',0):.4f}  "
              f"mAP_small={res.get('map_small',0):.4f}")
        print(f"  mAR@100={res.get('mar_100',0):.4f}")

        print(f"  Classification on matched detections ({sn})...")
        gt_a, pred_a, n_gt, n_pred, n_m = evaluate_classification(
            model, loader, device)
        cls_data[sn] = {
            "gt": gt_a, "pred": pred_a,
            "n_gt": n_gt, "n_pred": n_pred, "n_matched": n_m,
        }
        acc = (gt_a == pred_a).mean() if len(gt_a) > 0 else 0
        print(f"  GT:{n_gt}  Preds:{n_pred}  Matched:{n_m}  Acc:{acc:.4f}")

    for sn in ["val", "test"]:
        cd = cls_data[sn]
        gt, pred = cd["gt"], cd["pred"]
        if len(gt) == 0:
            continue
        print(f"\n{'='*60}")
        print(f"Classification Report (Matched Detections) - {sn}")
        print(f"{'='*60}")
        present = sorted(set(gt.tolist()) | set(pred.tolist()))
        names   = [CLASS_NAMES[c - 1] if 1 <= c <= N_CLS else f"c{c}"
                   for c in present]
        print(classification_report(
            gt, pred, labels=present, target_names=names,
            digits=4, zero_division=0))

    for sn in ["val", "test"]:
        cd = cls_data[sn]
        gt, pred  = cd["gt"], cd["pred"]
        cd["acc"]     = (gt == pred).mean() if len(gt) > 0 else 0
        cd["f1m"]     = (f1_score(gt, pred, average="macro", zero_division=0)
                         if len(gt) > 0 else 0)
        cd["det_rec"] = cd["n_matched"] / max(cd["n_gt"], 1)

    print(f"\n{'='*70}")
    print(f"SUMMARY: ResNet-50 + Faster R-CNN + {head_label}")
    print(f"Training mode : {args.training_mode}")
    print(f"{'='*70}")
    print(f"{'Detection Metrics':<30s} {'Val':>12s} {'Test':>12s}")
    print("-" * 54)
    for lab, k in [("mAP@0.5:0.95", "map"), ("mAP@0.5", "map_50"),
                   ("mAP@0.75", "map_75"), ("mAP_small", "map_small"),
                   ("mAP_medium", "map_medium"), ("mAP_large", "map_large"),
                   ("mAR@100", "mar_100")]:
        vs = []
        for s in ["val", "test"]:
            v = final_results[s].get(k, 0)
            v = v.item() if isinstance(v, torch.Tensor) and v.numel() == 1 else v
            vs.append(max(v, 0))
        print(f"  {lab:<28s} {vs[0]:>12.4f} {vs[1]:>12.4f}")

    print(f"\n{'Classification (Matched)':<30s} {'Val':>12s} {'Test':>12s}")
    print("-" * 54)
    for lab, k in [("Cls Accuracy", "acc"), ("Macro F1", "f1m"),
                   ("Det Recall", "det_rec")]:
        vs = [cls_data[s].get(k, 0) for s in ["val", "test"]]
        print(f"  {lab:<28s} {vs[0]:>12.4f} {vs[1]:>12.4f}")

    print(f"\nBackbone  : ResNet-50-FPN (COCO pretrained)")
    print(f"Framework : Faster R-CNN (RPN + ROI pooling, COCO pretrained)")
    print(f"Head      : {head_label}")
    print(f"Mode      : {args.training_mode}  ({trainable_desc})")
    print(f"Params    : {n_p:.2f}M total, roi_head={n_head:.4f}M, "
          f"trained={n_train:.3f}M")
    print(f"Training  : {total_epochs_trained} total epochs  "
          f"(base={args.epochs}"
          + (f", finetune={args.finetune_epochs}"
             if args.training_mode == "end_to_end_then_finetune" else "")
          + ")")
    print(f"Best      : epoch {best_epoch}, mAP@0.5={best_map50:.4f}")

    run_plot_dir = PLOT_DIR / tag
    run_plot_dir.mkdir(parents=True, exist_ok=True)

    print("\nGenerating plots...")
    plot_training_curves(history, total_epochs_trained, run_plot_dir, head_label)
    plot_map_bar(final_results, run_plot_dir, head_label)
    plot_per_class_ap(final_results, run_plot_dir)
    plot_confusion_matrices(cls_data, run_plot_dir)
    plot_cls_accuracy(cls_data, run_plot_dir)
    with torch.no_grad():
        plot_sample_detections(model, val_ds, device, run_plot_dir)

    print("\nComputing per-class confidence curves...")
    per_class_curves = {}
    for sn, loader in [("val", val_loader), ("test", test_loader)]:
        print(f"  Collecting predictions for {sn}...")
        all_preds, all_gts = collect_all_predictions(model, loader, device)
        curves             = compute_per_class_curves(all_preds, all_gts)
        per_class_curves[sn] = curves
        plot_pr_curves_per_class(curves, run_plot_dir, sn)
        plot_f1_confidence_per_class(curves, run_plot_dir, sn)
        plot_precision_confidence_per_class(curves, run_plot_dir, sn)
        plot_recall_confidence_per_class(curves, run_plot_dir, sn)
        mean_ap_pr, _ = plot_ap_aucpr_bar(curves, run_plot_dir, sn)
        print(f"  {sn}: Mean AP (AUC-PR) @IoU=0.5 = {mean_ap_pr:.4f}")

    # ---- save metrics JSON ----
    def t2s(v):
        if isinstance(v, torch.Tensor):
            return v.item() if v.numel() == 1 else v.tolist()
        return v

    sd = {}
    for sn in ["val", "test"]:
        r  = final_results[sn]
        cd = cls_data[sn]
        sd[sn] = {
            "detection": {k: t2s(r.get(k, 0)) for k in
                          ["map", "map_50", "map_75", "map_small", "map_medium",
                           "map_large", "mar_100", "map_per_class"]},
            "classification": {
                "accuracy":   float(cd.get("acc", 0)),
                "macro_f1":   float(cd.get("f1m", 0)),
                "det_recall": float(cd.get("det_rec", 0)),
                "n_gt":       int(cd["n_gt"]),
                "n_matched":  int(cd["n_matched"]),
            },
        }
    sd["training"] = {k: v for k, v in history.items()}
    for sn in ["val", "test"]:
        curves = per_class_curves.get(sn, {})
        sd[sn]["ap_aucpr_per_class"] = {
            CLASS_NAMES[cls - 1]: curves.get(cls, {}).get("ap", 0.0)
            for cls in range(1, N_CLS_WITH_BG)
        }
        aps = [curves.get(cls, {}).get("ap", 0.0)
               for cls in range(1, N_CLS_WITH_BG)]
        sd[sn]["mean_ap_aucpr"] = float(np.mean(aps))
    sd["config"] = {
        "model":            f"ResNet-50 + Faster R-CNN + {head_label}",
        "head_type":        args.head,
        "training_mode":    args.training_mode,
        "trainable_desc":   trainable_desc,
        "n_epochs":         args.epochs,
        "finetune_epochs":  args.finetune_epochs,
        "total_epochs":     total_epochs_trained,
        "lr_backbone":      args.lr_backbone,
        "lr_head":          args.lr_head,
        "img_max_size":     args.img_max_size,
        "dropout":          args.dropout,
        "batch_size":       args.batch_size,
        "best_epoch":       best_epoch,
        "best_map50":       best_map50,
        "params_total_M":   n_p,
        "params_roihead_M": n_head,
        "params_trained_M": n_train,
    }
    mp = run_plot_dir / f"metrics_frcnn_{tag}.json"
    with open(mp, "w") as f:
        json.dump(sd, f, indent=2, default=str)
    print(f"Metrics saved to {mp}")


if __name__ == "__main__":
    main()
