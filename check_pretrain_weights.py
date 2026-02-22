#!/usr/bin/env python
"""
Verify ImageNet pretrained weights download and can be loaded.
Run this BEFORE training to confirm weights are available.
"""

import torch
import torchvision.models as models
import os
from pathlib import Path

def check_pretrained_weights():
    """Check if pretrained weights download and are usable."""
    
    print("\n" + "="*70)
    print("CHECKING IMAGENET PRETRAINED WEIGHTS")
    print("="*70)
    
    # Get torch cache directory
    torch_home = os.path.expanduser(os.getenv('TORCH_HOME', '~/.cache/torch'))
    cache_dir = os.path.join(torch_home, 'hub', 'checkpoints')
    
    print(f"\n📁 PyTorch cache directory: {cache_dir}")
    print(f"   Exists: {os.path.exists(cache_dir)}")
    
    if os.path.exists(cache_dir):
        files = os.listdir(cache_dir)
        print(f"   Files: {files}")
    
    # Try loading ResNet models
    models_to_test = {
        'resnet18': models.resnet18,
        'resnet34': models.resnet34,
        'resnet50': models.resnet50,
    }
    
    print("\n" + "-"*70)
    print("DOWNLOADING MODELS (first time takes 1-2 min each)")
    print("-"*70)
    
    for name, model_fn in models_to_test.items():
        try:
            print(f"\n⏳ Loading {name}...", end=" ", flush=True)
            
            if name == 'resnet18':
                model = model_fn(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            elif name == 'resnet34':
                model = model_fn(weights=models.ResNet34_Weights.IMAGENET1K_V1)
            elif name == 'resnet50':
                model = model_fn(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            
            # Count parameters
            num_params = sum(p.numel() for p in model.parameters())
            state_dict_size = len(model.state_dict())
            
            print(f"✅ SUCCESS")
            print(f"   Parameters: {num_params:,}")
            print(f"   State dict keys: {state_dict_size}")
            
        except Exception as e:
            print(f"❌ FAILED")
            print(f"   Error: {str(e)}")
    
    # List cached files
    print("\n" + "-"*70)
    print("CACHED WEIGHTS IN FILESYSTEM")
    print("-"*70)
    
    if os.path.exists(cache_dir):
        for f in sorted(os.listdir(cache_dir)):
            fpath = os.path.join(cache_dir, f)
            size_mb = os.path.getsize(fpath) / (1024*1024)
            print(f"   {f:50s} {size_mb:6.1f} MB")
    else:
        print(f"   Cache directory doesn't exist yet: {cache_dir}")
    
    print("\n" + "="*70)
    print("✅ IF ALL MODELS LOADED, YOUR WEIGHTS ARE WORKING!")
    print("="*70 + "\n")

if __name__ == "__main__":
    check_pretrained_weights()
