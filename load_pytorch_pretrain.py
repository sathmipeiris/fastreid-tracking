#!/usr/bin/env python
"""
Manual pretrained weight loader for ReID models.
Use this if automatic download fails.

This directly loads PyTorch's official pretrained weights
before training starts.
"""

import torch
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def load_torchvision_pretrained(model, model_name='resnet18'):
    """
    Manually load PyTorch pretrained weights into ReID model.
    
    Args:
        model: The ReID model to load weights into
        model_name: resnet18, resnet34, resnet50, etc.
    
    Returns:
        model with pretrained weights loaded
    """
    try:
        import torchvision.models as models
        
        logger.info(f"Loading official PyTorch pretrained {model_name}...")
        
        # Load official pretrained model
        if model_name == 'resnet18':
            pretrained_model = models.resnet18(pretrained=True)
        elif model_name == 'resnet34':
            pretrained_model = models.resnet34(pretrained=True)
        elif model_name == 'resnet50':
            pretrained_model = models.resnet50(pretrained=True)
        elif model_name == 'resnet101':
            pretrained_model = models.resnet101(pretrained=True)
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        # Extract backbone state dict (ignore classification layer)
        pretrained_dict = pretrained_model.state_dict()
        
        # Load into ReID backbone
        model_dict = model.state_dict()
        
        # Filter only matching keys
        matched_keys = {}
        for k, v in pretrained_dict.items():
            if k in model_dict and v.shape == model_dict[k].shape:
                matched_keys[k] = v
        
        logger.info(f"Loading {len(matched_keys)} matching weights from PyTorch pretrained {model_name}")
        
        # Load weights
        incompatible = model.load_state_dict(matched_keys, strict=False)
        
        if incompatible.missing_keys:
            logger.warning(f"Missing keys in ReID model: {len(incompatible.missing_keys)}")
        if incompatible.unexpected_keys:
            logger.warning(f"Unexpected keys in pretrained model: {len(incompatible.unexpected_keys)}")
        
        logger.info("✓ Pretrained weights loaded successfully!")
        return model
        
    except Exception as e:
        logger.error(f"Failed to load pretrained weights: {str(e)}")
        raise

if __name__ == "__main__":
    # Test: Create a dummy model and load weights
    from fastreid.config import get_cfg
    from fastreid.engine import DefaultTrainer
    
    cfg = get_cfg()
    cfg.merge_from_file('custom_configs/resnet18_pytorch_pretrain.yml')
    
    model = DefaultTrainer.build_model(cfg)
    load_torchvision_pretrained(model, 'resnet18')
    
    print("✓ Weights loaded successfully!")
