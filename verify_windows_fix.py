#!/usr/bin/env python
"""
Verify that Windows PyTorch fix is properly applied
Run this script to check if the data_utils.py has been patched
"""
import os
import sys

print("=" * 80)
print("WINDOWS PYTORCH FIX VERIFICATION")
print("=" * 80)
print()

# Add FastReID to path
sys.path.insert(0, os.path.join(os.getcwd(), 'fast-reid'))

# Test 1: Check file modifications
print("[1] Checking data_utils.py modifications...")
data_utils_path = 'fast-reid/fastreid/data/data_utils.py'
try:
    with open(data_utils_path, 'r') as f:
        content = f.read()
    
    checks = {
        'Windows platform check': "sys.platform != 'win32'" in content,
        'CUDA availability check': "torch.cuda.is_available()" in content,
        'Stream None initialization': "self.stream = None" in content,
        'BackgroundGenerator skip': "if torch.cuda.is_available() and sys.platform != 'win32':" in content,
    }
    
    all_good = True
    for check_name, result in checks.items():
        status = "✓" if result else "✗"
        print(f"  {status} {check_name}: {'OK' if result else 'MISSING'}")
        all_good = all_good and result
    
    if not all_good:
        print("\n  ⚠️  WARNING: Some patches not found. File may not be modified correctly.")
    else:
        print("\n  ✓ All patches successfully applied!")
except Exception as e:
    print(f"  ✗ Error reading file: {e}")

print()

# Test 2: Check PyTorch
print("[2] Checking PyTorch installation...")
try:
    import torch
    print(f"  ✓ PyTorch version: {torch.__version__}")
    print(f"  ✓ CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  ✓ CUDA device: {torch.cuda.get_device_name(0)}")
except Exception as e:
    print(f"  ✗ Error: {e}")

print()

# Test 3: Check FastReID imports
print("[3] Checking FastReID imports...")
try:
    from fastreid.config import get_cfg
    print(f"  ✓ FastReID config loaded")
    from fastreid.data.data_utils import DataLoaderX
    print(f"  ✓ DataLoaderX imported successfully")
except Exception as e:
    print(f"  ✗ Error importing: {e}")

print()

# Test 4: Check environment variables
print("[4] Checking environment variables...")
env_vars = {
    'TORCH_HOME': os.environ.get('TORCH_HOME'),
    'KMP_DUPLICATE_LIB_OK': os.environ.get('KMP_DUPLICATE_LIB_OK'),
    'CUDA_LAUNCH_BLOCKING': os.environ.get('CUDA_LAUNCH_BLOCKING'),
    'PYTHONPATH': os.environ.get('PYTHONPATH'),
}

for var_name, var_value in env_vars.items():
    status = "✓" if var_value else "○"
    print(f"  {status} {var_name}: {var_value if var_value else '(not set)'}")

print()

# Test 5: Check config files
print("[5] Checking config files...")
config_files = [
    'custom_configs/base_with_eval.yml',
    'custom_configs/plateau_solutions/solution_5_smaller_batch_higher_lr.yml',
]

for config_file in config_files:
    try:
        with open(config_file, 'r') as f:
            content = f.read()
        
        has_num_workers = 'NUM_WORKERS' in content
        has_zero_workers = 'NUM_WORKERS: 0' in content
        
        status = "✓" if has_zero_workers else "✗" if has_num_workers else "○"
        print(f"  {status} {config_file}: {
            'NUM_WORKERS=0' if has_zero_workers else 
            'NUM_WORKERS set but not 0' if has_num_workers else 
            'NUM_WORKERS not specified (inherit)'
        }")
    except Exception as e:
        print(f"  ✗ {config_file}: Error - {e}")

print()
print("=" * 80)
print("VERIFICATION COMPLETE")
print("=" * 80)
print()

print("Next steps:")
print("  1. If all checks passed: Ready to train! Run: train_windows_recommended.bat")
print("  2. If any checks failed: Review WINDOWS_PYTORCH_FIX.md for details")
print("  3. Main fix: fast-reid/fastreid/data/data_utils.py has Windows compatibility")
print()
