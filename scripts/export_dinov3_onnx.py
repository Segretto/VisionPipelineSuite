#!/usr/bin/env python3
"""
Export DINOv3 Segmentation Model to ONNX

Usage:
    python scripts/export_dinov3_onnx.py \
        config=dinov3/dinov3/eval/segmentation/configs/config-ade20k-linear-training.yaml \
        model.config_file=dinov3/dinov3/configs/backbone_vitb.yaml \
        model.pretrained_weights=/home/segreto/Models/dinov3_vitb16_pretrain.pth \
        output_dir=./dinov3_seg_ft \
        export_path=./dinov3_seg_coco.onnx
"""

import sys
import os
import torch
import logging
from omegaconf import OmegaConf

# Add the dinov3 directory to the python path
current_dir = os.path.dirname(os.path.abspath(__file__))
dinov3_path = os.path.join(os.path.dirname(current_dir), "dinov3")
if dinov3_path not in sys.path:
    sys.path.append(dinov3_path)

from dinov3.eval.segmentation.models import build_segmentation_decoder
from dinov3.eval.setup import load_model_and_context
from dinov3.eval.helpers import cli_parser
import dinov3.distributed as distributed

logger = logging.getLogger("dinov3_export")
logging.basicConfig(level=logging.INFO)

def main():
    # Enable distributed mode (required for config setup)
    distributed.enable(overwrite=True)

    # 1. Parse Config
    # We use the same CLI parser as training to ensure compatibility
    # But we need to handle the case where we just want to export, not train
    
    # Manually parse args to get config file and overrides
    args = sys.argv[1:]
    
    # Filter out export_path if present, as it's not in the DINOv3 config
    export_path = "dinov3_seg_coco.onnx"
    dinov3_args = []
    for arg in args:
        if arg.startswith("export_path="):
            export_path = arg.split("=")[1]
        else:
            dinov3_args.append(arg)
            
    # Load base config
    # We can use a default config structure and merge
    # For simplicity, we'll assume the user provides the same config as training
    
    # Use DINOv3's cli_parser to get the merged config
    # We might need to mock some things if cli_parser expects specific args
    # But looking at helpers.py, it just uses OmegaConf.from_cli
    
    # Let's construct the config manually to be safe and flexible
    # 1. Load the base yaml if provided in args
    base_config = None
    for arg in dinov3_args:
        if arg.startswith("config="):
            config_path = arg.split("=")[1]
            base_config = OmegaConf.load(config_path)
            break
            
    if base_config is None:
        logger.error("Please provide a base config using config=...")
        sys.exit(1)
        
    # 2. Merge CLI args
    cli_config = OmegaConf.from_cli(dinov3_args)
    config = OmegaConf.merge(base_config, cli_config)

    # Ensure config.model exists and has defaults
    if "model" not in config:
        config.model = OmegaConf.create({})
    if "dino_hub" not in config.model:
        config.model.dino_hub = None
    if "config_file" not in config.model:
        config.model.config_file = None
    if "pretrained_weights" not in config.model:
        config.model.pretrained_weights = None
    
    # 2. Build Model (Backbone + Head)
    logger.info("Building model...")
    
    # Load Backbone
    # This uses model.config_file and model.pretrained_weights
    backbone, _ = load_model_and_context(config.model, config.output_dir)
    
    # Build Segmentation Decoder (Head)
    # This creates the FeatureDecoder which wraps [backbone, decoder]
    segmentation_model = build_segmentation_decoder(
        backbone,
        config.decoder_head.backbone_out_layers,
        config.decoder_head.type, # "linear"
        num_classes=config.decoder_head.num_classes,
        autocast_dtype=torch.float32, # Force float32 for export
        dropout=config.decoder_head.dropout,
    )
    
    # 3. Load Fine-Tuned Weights
    # The checkpoint contains "model" key which has keys like "segmentation_model.1.conv.weight"
    # The backbone (segmentation_model.0) was frozen, so it's not in the checkpoint usually
    # But we loaded the backbone from pretrained_weights above, so that's fine.
    
    checkpoint_path = os.path.join(config.output_dir, "model_final.pth")
    if not os.path.exists(checkpoint_path):
        logger.error(f"Checkpoint not found at {checkpoint_path}")
        sys.exit(1)
        
    logger.info(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    # The checkpoint keys are likely prefixed with "module." if trained with DDP
    # And they only contain the head (segmentation_model.1)
    state_dict = checkpoint["model"]
    
    # We need to load this into segmentation_model
    # The segmentation_model is a FeatureDecoder, which has self.segmentation_model = nn.ModuleList([backbone, decoder])
    # So segmentation_model.1 is the decoder.
    
    # Clean up state_dict keys if needed
    new_state_dict = {}
    for k, v in state_dict.items():
        # Remove "module." prefix if present (from DDP)
        if k.startswith("module."):
            k = k[7:]
        new_state_dict[k] = v
        
    # Load weights
    # strict=False because we might be missing backbone weights in the checkpoint (which is expected)
    # But we must ensure the head weights match
    missing_keys, unexpected_keys = segmentation_model.load_state_dict(new_state_dict, strict=False)
    
    logger.info(f"Missing keys: {len(missing_keys)}")
    logger.info(f"Unexpected keys: {len(unexpected_keys)}")
    
    # Verify that we loaded the head weights
    # Missing keys should be all backbone keys (segmentation_model.0...)
    # We should NOT have missing keys for segmentation_model.1
    head_missing = [k for k in missing_keys if "segmentation_model.1" in k]
    if head_missing:
        logger.error(f"Error: Failed to load head weights. Missing: {head_missing}")
        sys.exit(1)
        
    logger.info("Successfully loaded fine-tuned weights.")
    
    # 4. Prepare for Export
    segmentation_model.eval()
    segmentation_model.to("cpu") # Export on CPU
    
    # Create dummy input
    # DINOv3 typically takes (B, 3, H, W)
    # Let's use 512x512 as default, but make it dynamic
    dummy_input = torch.randn(1, 3, 512, 512)
    
    logger.info(f"Exporting to ONNX at {export_path}...")
    
    # Dynamic axes
    dynamic_axes = {
        "input": {0: "batch_size", 2: "height", 3: "width"},
        "output": {0: "batch_size", 2: "height", 3: "width"}
    }
    
    torch.onnx.export(
        segmentation_model,
        dummy_input,
        export_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
        opset_version=17, # Use a recent opset
        do_constant_folding=True,
    )
    
    logger.info("ONNX export complete!")

if __name__ == "__main__":
    main()
