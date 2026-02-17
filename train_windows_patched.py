#!/usr/bin/env python
"""
Windows-compatible PyTorch fix - patches FastReID for safe Windows training
This disables problematic CUDA tensor device indexing and background threading
"""
import os
import sys

# Must be set BEFORE importing PyTorch or FastReID
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_MAX_THREADS'] = '1'
os.environ['TORCH_HOME'] = os.path.join(os.getcwd(), '.torch_cache')
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['PYTHONPATH'] = os.path.join(os.getcwd(), 'fast-reid') + os.pathsep + os.environ.get('PYTHONPATH', '')
os.environ['FASTREID_DATASETS'] = os.path.join(os.getcwd(), 'fast-reid', 'datasets')

import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

# Monkey-patch FastReID's DataLoaderX to fix Windows multiprocessing issues
def patch_dataloader_for_windows():
    """Patches DataLoaderX to avoid multiprocessing issues on Windows"""
    try:
        from fastreid.data.data_utils import DataLoaderX, BackgroundGenerator
        import types
        
        original_init = DataLoaderX.__init__
        original_iter = DataLoaderX.__iter__
        
        def patched_init(self, local_rank, **kwargs):
            """Skip CUDA stream creation on Windows or when CUDA unavailable"""
            # Don't use CUDA streams on Windows
            self.local_rank = local_rank
            self.stream = None
            if torch.cuda.is_available() and sys.platform != 'win32':
                self.stream = torch.cuda.Stream(local_rank)
            torch.utils.data.DataLoader.__init__(self, **kwargs)
        
        def patched_iter(self):
            """Simplified iteration without background generator on Windows"""
            if sys.platform == 'win32' or not torch.cuda.is_available():
                # On Windows, just use regular DataLoader iteration
                return torch.utils.data.DataLoader.__iter__(self)
            else:
                # On other platforms with CUDA, use background generator
                self.iter = torch.utils.data.DataLoader.__iter__(self)
                self.iter = BackgroundGenerator(self.iter, self.local_rank)
                self.preload()
                return self
        
        DataLoaderX.__init__ = patched_init
        DataLoaderX.__iter__ = patched_iter
        
        print("[PATCH] DataLoaderX patched for Windows compatibility")
        return True
    except Exception as e:
        print(f"[WARNING] Could not patch DataLoaderX: {e}")
        return False

# Apply the patch
patch_dataloader_for_windows()

# Now run the training
if __name__ == "__main__":
    import subprocess
    
    config_file = 'custom_configs/plateau_solutions/solution_5_smaller_batch_higher_lr.yml'
    if len(sys.argv) > 1:
        config_file = sys.argv[1]
    
    output_dir = 'logs/market1501/plateau_solver'
    if len(sys.argv) > 2:
        output_dir = sys.argv[2]
    
    cmd = [
        sys.executable,
        'fast-reid/tools/train_net.py',
        '--config-file', config_file,
        '--num-gpus', '1',
        'OUTPUT_DIR', output_dir,
        'DATALOADER.NUM_WORKERS', '0',
        'SEED', '1234',
    ]
    
    print("=" * 80)
    print("WINDOWS PYTORCH FIX - FastReID Training")
    print("=" * 80)
    print(f"Config: {config_file}")
    print(f"Output: {output_dir}")
    print(f"Working directory: {os.getcwd()}")
    print(f"Python: {sys.executable}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.cuda.is_available()}")
    print("=" * 80)
    print()
    
    try:
        result = subprocess.run(cmd, check=False)
        sys.exit(result.returncode)
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error running training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
