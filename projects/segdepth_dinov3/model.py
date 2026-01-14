import torch
import torch.nn as nn
from backbone import DinoBackbone
from depth_head import DepthHeadLite
from seg_head import ASPPDecoder

class UnifiedDINOv3(nn.Module):
    def __init__(self, dino_model, num_classes, n_layers=12, embed_dim=384, depth_out_size=(640, 640), seg_target_size=(320, 320)):
        """
        dino_model: The pre-loaded DINOv3 model (e.g. via torch.hub)
        num_classes: Number of segmentation classes
        n_layers: Number of layers in DINO backbone to extract feature from (usually 12 for vits)
        embed_dim: Embedding dimension of DINO backbone (384 for vits)
        depth_out_size: Output resolution for depth map
        seg_target_size: Output resolution for segmentation map
        """
        super().__init__()
        self.backbone = DinoBackbone(dino_model, n_layers=n_layers)
        
        self.depth_head = DepthHeadLite(in_ch=embed_dim, out_size=depth_out_size)
        
        self.seg_head = ASPPDecoder(num_classes=num_classes, in_ch=embed_dim, target_size=seg_target_size)

    def forward(self, x):
        # Backbone forward
        feat = self.backbone(x) # (B, C, Hf, Wf)
        
        # Heads forward
        depth_out = self.depth_head(feat)
        seg_out = self.seg_head(feat)
        
        return depth_out, seg_out
