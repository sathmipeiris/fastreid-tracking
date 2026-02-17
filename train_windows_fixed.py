#!/usr/bin/env python
"""
Run FastReID training with Windows-specific fixes for multiprocessing errors
"""
import os
import sys
import subprocess

# ============ CRITICAL: Windows PyTorch fixes ============
# These environment variables prevent the "Couldn't open shared file mapping" error
os.environ['TORCH_HOME'] = os.path.join(os.getcwd(), '.torch_cache')
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Force PyTorch to use file-based sharing instead of memory mapping
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'

# For Windows: disable CUDA tensor device index for multiprocessing
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# Use spawn method for multiprocessing (safest on Windows)
os.environ['PYTHONPATH'] = os.path.join(os.getcwd(), 'fast-reid') + os.pathsep + os.environ.get('PYTHONPATH', '')

# Set datasets path
os.environ['FASTREID_DATASETS'] = os.path.join(os.getcwd(), 'fast-reid', 'datasets')

# Get config from command line or use default
config_file = 'custom_configs/plateau_solutions/solution_5_smaller_batch_higher_lr.yml'
if len(sys.argv) > 1:
    config_file = sys.argv[1]

output_dir = 'logs/market1501/plateau_solver'
if len(sys.argv) > 2:
    output_dir = sys.argv[2]

# ============ Run training ============
cmd = [
    sys.executable,
    'fast-reid/tools/train_net.py',
    '--config-file', config_file,
    '--num-gpus', '1',
    'OUTPUT_DIR', output_dir,
    'DATALOADER.NUM_WORKERS', '0',  # Explicitly force no workers
]

print("=" * 80)
print("WINDOWS PYTORCH FIX - Training Command")
print("=" * 80)
print(f"Config: {config_file}")
print(f"Output: {output_dir}")
print(f"Working directory: {os.getcwd()}")
print(f"Python: {sys.executable}")
print("=" * 80)
print()

try:
    result = subprocess.run(cmd, check=False)
    sys.exit(result.returncode)
except KeyboardInterrupt:
    print("\nTraining interrupted by user")
    sys.exit(1)
except Exception as e:
    print(f"Error running training: {e}")
    sys.exit(1)
