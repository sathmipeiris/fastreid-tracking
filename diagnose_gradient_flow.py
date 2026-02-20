#!/usr/bin/env python
"""
Diagnostic script to check if gradients are flowing through the model.
"""

import sys
import os
os.chdir(r'c:\Users\sathmi\Desktop\REID_TRAINING')
sys.path.insert(0, 'fast-reid')

import torch
from fastreid.config import get_cfg
from fastreid.engine import default_setup
from fastreid.modeling import build_model

# Load config
cfg = get_cfg()
cfg.merge_from_file('custom_configs/base_with_eval.yml')
cfg.SOLVER.MAX_EPOCH = 4
cfg.SOLVER.BASE_LR = 0.001
cfg.OUTPUT_DIR = 'logs/market1501/gradient_diagnostic'
default_setup(cfg)

# Build model
print("\n" + "="*80)
print("GRADIENT FLOW DIAGNOSTIC")
print("="*80 + "\n")

model = build_model(cfg)
model.train()  # Set to training mode!
print(f"Model training mode: {model.training}")

# Check which parameters have requires_grad=True
print("\n[1] CHECKING requires_grad STATUS:\n")
frozen_count = 0
trainable_count = 0
total_params = 0

for name, param in model.named_parameters():
    total_params += 1
    if not param.requires_grad:
        frozen_count += 1
        if frozen_count <= 5:  # Print first 5
            print(f"  ❌ FROZEN: {name}")
    else:
        trainable_count += 1
        if trainable_count <= 5:  # Print first 5
            print(f"  ✓ TRAINABLE: {name}")

print(f"\nTotal: {trainable_count} trainable, {frozen_count} frozen (out of {total_params})")

# Simulate a forward-backward pass
print("\n[2] SIMULATING FORWARD-BACKWARD PASS:\n")

# Create dummy input
dummy_input = torch.randn(4, 3, 256, 128).cuda()  # Batch of 4 images
dummy_labels = torch.randint(0, 751, (4,)).cuda()  # Random labels (0-750)

try:
    # Forward pass
    print(f"Input shape: {dummy_input.shape}")
    print(f"Input requires_grad: {dummy_input.requires_grad}")
    
    output = model(dummy_input)
    print(f"\nModel output type: {type(output)}")
    
    if isinstance(output, dict):
        print(f"Output keys: {output.keys()}")
        for key, val in output.items():
            if isinstance(val, torch.Tensor):
                print(f"  {key}: shape={val.shape}, requires_grad={val.requires_grad}")
    elif isinstance(output, torch.Tensor):
        print(f"Output shape: {output.shape}, requires_grad: {output.requires_grad}")
    
    # Check gradients after forward pass (before backward)
    print("\n[3] CHECKING GRADIENTS BEFORE BACKWARD:\n")
    has_grads_before = False
    for name, param in model.named_parameters():
        if param.grad is not None:
            has_grads_before = True
            print(f"  {name}: grad = {param.grad.mean().item():.6f}")
    
    if not has_grads_before:
        print("  (No gradients yet - expected before backward)")
    
    # Create a simple loss (MSE)
    print("\n[4] COMPUTING LOSS:\n")
    # Get fc output (assume it's in the model somewhere)
    if isinstance(output, dict) and 'cls' in output:
        logits = output['cls']
    elif isinstance(output, torch.Tensor):
        logits = output
    else:
        logits = list(output.values())[0] if output else None
    
    if logits is not None:
        print(f"Logits shape: {logits.shape}")
        print(f"Logits requires_grad: {logits.requires_grad}")
        
        # Simple CE loss
        loss = torch.nn.functional.cross_entropy(logits, dummy_labels)
        print(f"Loss value: {loss.item():.6f}")
        print(f"Loss requires_grad: {loss.requires_grad}")
        
        # Backward pass
        print("\n[5] RUNNING BACKWARD PASS:\n")
        loss.backward()
        print("Backward completed!")
        
        # Check gradients after backward
        print("\n[6] CHECKING GRADIENTS AFTER BACKWARD:\n")
        has_grads_after = False
        layers_with_grads = []
        layers_without_grads = []
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                if param.grad is not None:
                    has_grads_after = True
                    grad_norm = param.grad.abs().mean().item()
                    if grad_norm > 1e-7:
                        layers_with_grads.append((name, grad_norm))
                else:
                    layers_without_grads.append(name)
        
        if layers_with_grads:
            print(f"✓ Parameters with NON-ZERO gradients ({len(layers_with_grads)}):")
            for name, grad_norm in layers_with_grads[:10]:
                print(f"  {name}: avg grad = {grad_norm:.2e}")
        else:
            print("❌ NO PARAMETERS HAVE GRADIENTS!")
        
        if layers_without_grads:
            print(f"\n❌ Parameters with ZERO/None gradients ({len(layers_without_grads)}):")
            for name in layers_without_grads[:10]:
                print(f"  {name}")
    else:
        print("Could not find logits in output!")
        
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*80)
print("END DIAGNOSTIC")
print("="*80 + "\n")
