#!/usr/bin/env python
"""
Run FastReID training with the specified configuration
"""
import os
import sys
import subprocess

# Set environment variables
os.environ['FASTREID_DATASETS'] = 'fast-reid/datasets'
os.environ['PYTHONPATH'] = os.path.join(os.getcwd(), 'fast-reid') + os.pathsep + os.environ.get('PYTHONPATH', '')

# Run training command
cmd = [
    sys.executable,
    'fast-reid/tools/train_net.py',
    '--config-file', 'custom_configs/bagtricks_R50-ibn.yml',
    'OUTPUT_DIR', 'logs/market_r50_ibn'
]

print(f"Running: {' '.join(cmd)}")
print(f"Working directory: {os.getcwd()}")
print(f"FASTREID_DATASETS: {os.environ['FASTREID_DATASETS']}")

try:
    result = subprocess.run(cmd, check=False)
    sys.exit(result.returncode)
except KeyboardInterrupt:
    print("\nTraining interrupted by user")
    sys.exit(1)
except Exception as e:
    print(f"Error running training: {e}")
    sys.exit(1)
