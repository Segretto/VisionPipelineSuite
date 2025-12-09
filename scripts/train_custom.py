#!/usr/bin/env python3
"""
Custom Training Script for Instance Segmentation using DINOv3 Backbone (Transformers)

Usage:
    python scripts/train_custom.py \
        --model-name facebook/dinov3-vit-base-pretrain-lvd1689m \
        --train-images /home/segreto/Data/embed/images/train \
        --train-labels /home/segreto/Data/embed/labels/train/coco.json \
        --val-images /home/segreto/Data/embed/images/val \
        --val-labels /home/segreto/Data/embed/labels/val/coco.json \
        --output-dir ./output_custom_seg \
        --epochs 10 --batch-size 2
"""

import argparse
import os
import sys
import torch
import torch.utils.data
from torch import nn
import torchvision
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from torchvision.ops import MultiScaleRoIAlign
from transformers import AutoModel, AutoImageProcessor

# Add src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(os.path.dirname(current_dir), "src")
if src_path not in sys.path:
    sys.path.append(src_path)

from vision_suite.data.coco_instance_dataset import CocoInstanceDataset

def get_args():
    parser = argparse.ArgumentParser(description="Train DINOv3 Instance Segmentation")
    parser.add_argument("--model-name", type=str, default="facebook/dinov2-base", help="Hugging Face model ID")
    parser.add_argument("--train-images", type=str, required=True)
    parser.add_argument("--train-labels", type=str, required=True)
    parser.add_argument("--val-images", type=str, required=True)
    parser.add_argument("--val-labels", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="./output_custom_seg")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--num-classes", type=int, default=91) # COCO default
    return parser.parse_args()

class BackboneWrapper(nn.Module):
    def __init__(self, backbone, out_channels):
        super().__init__()
        self.backbone = backbone
        self.out_channels = out_channels
        
    def forward(self, x):
        # x is (B, 3, H, W) normalized tensor
        
        # Transformers model expects (B, 3, H, W)
        outputs = self.backbone(pixel_values=x)
        
        # last_hidden_state is (B, L, D)
        # L = H*W/P^2 + 1 (CLS) + registers
        last_hidden_state = outputs.last_hidden_state
        
        B, L, D = last_hidden_state.shape
        
        # Assuming patch size 14 (DINOv2 default) or 16
        # We need to infer H, W from L
        # Or simpler: use the input image size and patch size
        # DINOv2 usually uses patch size 14
        patch_size = 14 # Default for DINOv2 base
        
        # Check config if available
        if hasattr(self.backbone.config, "patch_size"):
            patch_size = self.backbone.config.patch_size
            
        # Remove CLS token (index 0)
        # Also check for registers (DINOv2 usually has 4 registers if enabled, but standard might not)
        # Standard DINOv2 from HF usually has 1 CLS token and no registers unless specified
        
        # Let's assume 1 CLS token for now
        patch_tokens = last_hidden_state[:, 1:, :]
        
        # Reshape to (B, D, H, W)
        # We need to know H, W of the feature map
        # H_feat = H_img // patch_size
        # W_feat = W_img // patch_size
        
        H_img, W_img = x.shape[-2:]
        h, w = H_img // patch_size, W_img // patch_size
        
        # Verify shape
        if patch_tokens.shape[1] != h * w:
            # Try to handle registers if present
            # If L > h*w + 1, maybe registers
            pass
            
        x_patch = patch_tokens.transpose(1, 2).reshape(B, D, h, w)
        
        return {"0": x_patch}

def collate_fn(batch):
    return tuple(zip(*batch))

# Simple transform wrapper to handle (img, target) signature
class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

class ToTensor(object):
    def __call__(self, image, target):
        image = torchvision.transforms.functional.to_tensor(image)
        return image, target

def get_transform(train):
    transforms = [ToTensor()]
    return Compose(transforms)

def main():
    args = get_args()
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    # 1. Load Backbone from Transformers
    print(f"Loading backbone {args.model_name}...")
    backbone_model = AutoModel.from_pretrained(args.model_name)
    print("Backbone loaded.")
    
    # Freeze backbone
    for param in backbone_model.parameters():
        param.requires_grad = False
        
    # Wrap backbone
    out_channels = backbone_model.config.hidden_size
    backbone = BackboneWrapper(backbone_model, out_channels)
    
    # 2. Create Mask R-CNN
    # We need to define anchor generator and roi pooler matching the backbone output
    
    anchor_generator = AnchorGenerator(
        sizes=((32, 64, 128, 256, 512),),
        aspect_ratios=((0.5, 1.0, 2.0),) * 5
    )
    
    roi_pooler = MultiScaleRoIAlign(
        featmap_names=['0'],
        output_size=7,
        sampling_ratio=2
    )
    
    mask_roi_pooler = MultiScaleRoIAlign(
        featmap_names=['0'],
        output_size=14,
        sampling_ratio=2
    )
    
    model = MaskRCNN(
        backbone,
        num_classes=args.num_classes,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler,
        mask_roi_pool=mask_roi_pooler
    )
    
    model.to(device)
    
    # 3. Dataset and DataLoader
    # We use standard ToTensor which converts to [0, 1]
    # DINOv2 expects normalized images (mean/std)
    # But for simplicity, we'll let the backbone handle it or add normalization
    # Ideally we should add Normalize transform
    
    # Let's add normalization to match DINOv2
    # mean = [0.485, 0.456, 0.406]
    # std = [0.229, 0.224, 0.225]
    
    class Normalize(object):
        def __call__(self, image, target):
            image = torchvision.transforms.functional.normalize(
                image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )
            return image, target

    def get_transform_norm(train):
        transforms = [ToTensor(), Normalize()]
        return Compose(transforms)
    
    dataset = CocoInstanceDataset(args.train_images, args.train_labels, get_transform_norm(train=True))
    dataset_val = CocoInstanceDataset(args.val_images, args.val_labels, get_transform_norm(train=False))
    
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=4,
        collate_fn=collate_fn
    )
    
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, batch_size=1, shuffle=False, num_workers=4,
        collate_fn=collate_fn
    )
    
    # 4. Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=0.0005)
    
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    
    # 5. Training Loop
    print("Starting training...")
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
    for epoch in range(args.epochs):
        model.train()
        i = 0
        for images, targets in data_loader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            
            if i % 10 == 0:
                print(f"Epoch: {epoch}, Iter: {i}, Loss: {losses.item():.4f}")
            i += 1
            
        lr_scheduler.step()
        
        # Save checkpoint
        torch.save(model.state_dict(), os.path.join(args.output_dir, f"model_{epoch}.pth"))
        print(f"Saved checkpoint to {args.output_dir}/model_{epoch}.pth")
        
    print("Training complete.")

if __name__ == "__main__":
    main()
