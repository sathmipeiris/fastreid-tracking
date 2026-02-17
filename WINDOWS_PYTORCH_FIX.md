# Windows PyTorch Multiprocessing Fix - FastReID

## Problem Summary

You were encountering this error:
```
RuntimeError: Couldn't open shared file mapping: <torch_19904_349185365_0>, error code: <1455>
```

This is a **Windows-specific PyTorch issue** with multiprocessing and shared memory mapping, typically occurring in:
- Data loading with multiple workers
- CUDA tensor device index synchronization
- Background thread/process interactions in DataLoaders

## Root Cause

FastReID uses a custom `DataLoaderX` class that:
1. Creates CUDA streams unconditionally (problems on Windows or without proper CUDA)
2. Uses `BackgroundGenerator` for prefetching (causes multiprocessing issues on Windows)
3. Doesn't handle Windows multiprocessing limitations

Error code 1455 (`ERROR_ALREADY_EXISTS`) indicates conflicts in temporary file/memory mapping creation.

## Solutions Applied

### 1. **Direct Code Fix (Recommended)** ✓
Modified `fast-reid/fastreid/data/data_utils.py`:
- Added Windows detection and CUDA availability checks
- Only creates CUDA streams when available and not on Windows
- Skips background generator on Windows
- Safely handles stream operations

**Files modified:**
- `fast-reid/fastreid/data/data_utils.py` - DataLoaderX class

### 2. **Environment Configuration**
Set critical environment variables:
```
TORCH_HOME=.torch_cache
KMP_DUPLICATE_LIB_OK=True
CUDA_LAUNCH_BLOCKING=1
OMP_NUM_THREADS=1
```

### 3. **DataLoader Configuration**
Ensure in all configs:
```yaml
DATALOADER:
  NUM_WORKERS: 0  # Disable worker processes
```

## How to Run Training

### Option 1: Simple Batch File (Recommended)
```bat
train_windows_recommended.bat
```

### Option 2: Python Script with Monkeypatching
```bat
python train_windows_patched.py
```

### Option 3: Manual Command
```bat
set TORCH_HOME=.torch_cache
set KMP_DUPLICATE_LIB_OK=True
set CUDA_LAUNCH_BLOCKING=1
set OMP_NUM_THREADS=1
set PYTHONPATH=%CD%\fast-reid

python fast-reid/tools/train_net.py ^
    --config-file custom_configs/plateau_solutions/solution_5_smaller_batch_higher_lr.yml ^
    --num-gpus 1 ^
    OUTPUT_DIR logs/market1501/plateau_solver ^
    DATALOADER.NUM_WORKERS 0
```

## What Was Changed

### In `fast-reid/fastreid/data/data_utils.py`:

1. **`DataLoaderX.__init__`**
   - Added `sys.platform != 'win32'` check
   - Only creates CUDA stream if CUDA is available AND not Windows
   - Added exception handling

2. **`DataLoaderX.__iter__`**
   - Only uses BackgroundGenerator on non-Windows platforms with CUDA
   - Returns standard DataLoader iterator on Windows

3. **`DataLoaderX.preload`**
   - Safely handles stream operations only if stream exists
   - Handles tensor device movement gracefully

4. **`DataLoaderX.__next__`**
   - Only waits on stream if it exists
   - Only preloads when background generator is active

## Testing the Fix

To verify the fix works:
```bash
# This should now run without the shared file mapping error
train_windows_recommended.bat

# Monitor first few iterations:
# - Should load data without RuntimeError
# - May see PyTorch warnings (normal, will auto-suppress)
# - Training should proceed normally
```

## Configuration Notes

**Current config: `solution_5_smaller_batch_higher_lr.yml`**
```yaml
IMS_PER_BATCH: 8          # Small batch for stability
NUM_WORKERS: 0            # NO worker processes (Windows safe)
BASE_LR: 0.001            # Higher learning rate
```

## Performance Notes

- **Without workers**: Slightly slower data loading, but stable
- **No background generator**: Direct iteration, no threading issues
- **Recommended for Windows**: This is the safe approach for Windows PyTorch

## If You Still Get Errors

1. **Check Python version**: Use Python 3.9+ for better PyTorch support
   ```bash
   python --version
   ```

2. **Update PyTorch** (if using older version):
   ```bash
   pip install --upgrade torch torchvision
   ```

3. **Check CUDA** (if using GPU):
   ```bash
   python -c "import torch; print(torch.cuda.is_available())"
   ```

4. **Verify modifications**:
   ```bash
   # Check that data_utils.py has our changes
   findstr /N "sys.platform" fast-reid\fastreid\data\data_utils.py
   ```

## Long-term Solution

If upgrading PyTorch becomes an option, check release notes for Windows multiprocessing improvements.

---

**Status**: ✅ Windows multiprocessing fix applied to FastReID
**Last Updated**: 2026-02-16
**Files Modified**: 1 (fast-reid/fastreid/data/data_utils.py)
**Solution**: Direct code patch + environment configuration
