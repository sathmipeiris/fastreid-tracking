# Training on Google Colaboratory - Step-by-Step Guide

Complete guide for training this FastReID Person Re-Identification model on Google Colab (Free GPU).

---

## 📋 Prerequisites

- Google Account (free)
- GitHub Account (recommended, to clone repo)
- Dataset (Market-1501) - download locally first
- ~3-4 hours of training time (depending on config)

---

## 🚀 Step 1: Prepare Your Repository

### Option A: Clone from GitHub (Recommended)
```bash
git clone https://github.com/YOUR_USERNAME/REID_TRAINING.git
cd REID_TRAINING
```

### Option B: Upload to Drive First (Recommended for Reliability)
1. **Upload your repo ZIP to Google Drive** (drag & drop into Drive)
2. Go to [Google Colab](https://colab.research.google.com) and create a new notebook
3. Mount Drive and copy to `/content` (see Cell 1 in Step 2)

### Option C: Upload ZIP Directly to Colab (Faster, Temporary)
1. Go to [Google Colab](https://colab.research.google.com)
2. Click file icon → upload → select your REID_TRAINING.zip
3. Unzip it: `!unzip /content/REID_TRAINING.zip`

---

## 📦 Step 2: Setup Environment in Colab

### Create New Colab Notebook

Click "New Notebook" or go to [Google Colab](https://colab.research.google.com)

### Cell 1: Check GPU & Basic Setup
```python
# Check GPU availability
!nvidia-smi

# Check Python version
import sys
print(f"Python: {sys.version}")

# Mount Google Drive (optional, for saving models)
from google.colab import drive
drive.mount('/content/drive')
```

**Note:** The `fast-reid` folder is **already included** in your REID_TRAINING repo. 
Once you clone/copy REID_TRAINING, you automatically get fast-reid.

### Cell 2: Clone/Copy Your REID_TRAINING Repository (contains fast-reid)
```python
import os
import subprocess

os.chdir('/content')

# Clone repo
try:
    subprocess.run(['git', 'clone', 'https://github.com/YOUR_USERNAME/REID_TRAINING.git'], check=True)
    print("✓ Repository cloned from GitHub")
except Exception as e:
    print(f"✗ Clone failed: {e}")
    print("Try Option B (Copy from Drive) instead")

# Verify clone worked
if os.path.exists('REID_TRAINING'):
    os.chdir('/content/REID_TRAINING')
    print(f"✓ Working directory: {os.getcwd()}")
    print(f"Contents: {os.listdir('.')[:5]}...")  # Show first 5 items
else:
    print("✗ REID_TRAINING folder not found - clone failed!")
```

**OR if you uploaded ZIP to Drive, copy it instead:**
```python
import shutil
import os

os.chdir('/content')

# Copy from Drive
try:
    shutil.copytree('/content/drive/My Drive/REID_TRAINING', '/content/REID_TRAINING', dirs_exist_ok=True)
    print("✓ Repository copied from Drive")
    os.chdir('/content/REID_TRAINING')
    print(f"✓ Working directory: {os.getcwd()}")
except Exception as e:
    print(f"✗ Copy failed: {e}")
    print("Make sure you uploaded REID_TRAINING folder to Drive root")
```

### Cell 3: Install PyTorch for Colab
```python
!pip install --upgrade pip
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Cell 4: Install FastReID Dependencies (from fast-reid/docs/requirements.txt which is in your repo)
```python
import os
import subprocess

# Verify we're in the right directory
os.chdir('/content/REID_TRAINING')
print(f"Current directory: {os.getcwd()}")

# Check that fast-reid exists (it should - it's in your repo)
if not os.path.exists('fast-reid'):
    print("✗ ERROR: fast-reid folder not found!")
    print("Make sure your REID_TRAINING repo includes the fast-reid folder")
else:
    print("✓ fast-reid folder found")

# Install common dependencies
!pip install opencv-python faiss-cpu yacs termcolor tabulate cloudpickle tqdm wheel scikit-learn tensorboard

# Install from requirements.txt
if os.path.exists('fast-reid/docs/requirements.txt'):
    print("✓ Installing from fast-reid/docs/requirements.txt")
    subprocess.run(['pip', 'install', '-r', 'fast-reid/docs/requirements.txt'], check=True)
else:
    print("⚠ requirements.txt not found - main deps already installed")
```

### Cell 5: Install FastReID Package (in editable mode)
```python
import subprocess, sys

# Install fastreid from the cloned repo
subprocess.check_call([sys.executable, "-m", "pip", "install", "-e", "fast-reid"])

# Verify installation
try:
    import fastreid
    print(f"✓ FastReID installed successfully: {fastreid.__version__}")
except ImportError:
    print("⚠ FastReID import failed - may need debugging")
```

---

## 📊 Step 3: Prepare Dataset

### Option A: Download Market-1501 in Colab
```python
import os
from pathlib import Path

# Create dataset directory
dataset_dir = Path('/content/REID_TRAINING/fast-reid/datasets/Market-1501-v15.09.15')
dataset_dir.mkdir(parents=True, exist_ok=True)

# Download using gdown (if available) or wget
# Note: Market-1501 may require manual download due to size/auth
# Download from: https://www.lipreading.com/Market-1501/ and upload to Colab

print(f"Dataset directory ready: {dataset_dir}")
```

### Option B: Upload from Your Local Machine
```python
from google.colab import files

print("Select Market-1501 dataset files to upload...")
uploaded = files.upload()

# Unzip if needed
for filename in uploaded.keys():
    if filename.endswith('.zip'):
        !unzip "/content/{filename}" -d "/content/REID_TRAINING/fast-reid/datasets/"
```

### Option C: Use Google Drive (Recommended for Large Files)
```python
# If dataset is in your Google Drive
import shutil

drive_dataset = '/content/drive/My Drive/Market-1501-v15.09.15'
local_dataset = '/content/REID_TRAINING/fast-reid/datasets/Market-1501-v15.09.15'

if os.path.exists(drive_dataset):
    shutil.copytree(drive_dataset, local_dataset, dirs_exist_ok=True)
    print("✓ Dataset copied from Drive")
else:
    print("⚠ Dataset not found in Drive")
```

---

## 🏋️ Step 4: Start Training

### Cell 6: Set Environment Variables & Run Training
```python
import os
import sys

# Set Python path
os.environ['PYTHONPATH'] = '/content/REID_TRAINING/fast-reid'
os.environ['FASTREID_DATASETS'] = '/content/REID_TRAINING/fast-reid/datasets'

# Change to working directory
os.chdir('/content/REID_TRAINING')

# Verify setup
print(f"Working dir: {os.getcwd()}")
print(f"PYTHONPATH: {os.environ['PYTHONPATH']}")
```

### Cell 7: Run Training Script
```python
!python train_research_grade.py \
    --config-file custom_configs/bagtricks_R50-ibn.yml \
    --run-name colab_training

# Alternative: Use early stopping version
# !python train_with_early_stopping.py \
#     --config-file custom_configs/bagtricks_R50-ibn.yml \
#     OUTPUT_DIR logs/market1501/colab_training
```

### Cell 8: Monitor Training (Optional)
```python
# Launch TensorBoard
%tensorboard --logdir /content/REID_TRAINING/logs
```

---

## 💾 Step 5: Save Results to Google Drive

### Cell 9: Backup Model Weights
```python
import shutil
from pathlib import Path

# Define paths
local_logs = Path('/content/REID_TRAINING/logs')
drive_backup = Path('/content/drive/My Drive/REID_Models')
drive_backup.mkdir(exist_ok=True)

# Copy trained model
if local_logs.exists():
    for pth_file in local_logs.glob('**/model_final.pth'):
        dest = drive_backup / pth_file.parent.name
        dest.mkdir(exist_ok=True)
        shutil.copy(pth_file, dest / pth_file.name)
        print(f"✓ Saved: {pth_file.name}")

# Copy logs
logs_backup = drive_backup / 'training_logs'
logs_backup.mkdir(exist_ok=True)
shutil.copytree(local_logs, logs_backup, dirs_exist_ok=True)
print("✓ All results backed up to Google Drive")
```

---

## ⚙️ Configuration Options

### Quick Training (Faster, Less Accurate)
```bash
!python train_research_grade.py \
    --config-file custom_configs/test_quick_sanity_check.yml \
    --run-name quick_test
```

### Production Training (Better Results)
```bash
!python train_research_grade.py \
    --config-file custom_configs/plateau_solutions/solution_1_higher_lr.yml \
    --run-name solution_1_test
```

### Adjust Batch Size for Memory
Edit config and add:
```yaml
SOLVER:
  IMS_PER_BATCH: 32  # Reduce if OOM (out of memory)
```

---

## 🔧 Troubleshooting

### Out of Memory (OOM) Error
```python
# Reduce batch size in config
# Or restart kernel and try again

import torch
torch.cuda.empty_cache()
```

### CUDA Memory Issues
```python
# Check memory
!nvidia-smi

# Clear cache
import gc
gc.collect()
```

### Dataset Not Found
```python
# Verify dataset structure
!ls -la /content/REID_TRAINING/fast-reid/datasets/
```

### Module Import Errors
```python
# Reinstall package
!pip install --force-reinstall -e /content/REID_TRAINING/fast-reid
```

---

## 📈 Expected Training Times

| GPU Type | Batch Size | Time for 120 Epochs |
|----------|-----------|-------------------|
| Tesla T4 (Free) | 64 | 8-10 hours |
| Tesla P100 | 64 | 5-6 hours |
| Tesla V100 | 128 | 3-4 hours |
| Tesla A100 (Pro) | 128 | 1.5-2 hours |

---

## ✅ After Training

### Download Results
```python
from google.colab import files
import os

# Download best model
model_path = '/content/REID_TRAINING/logs/market1501/colab_training/model_final.pth'
if os.path.exists(model_path):
    files.download(model_path)
```

### Evaluate Model
```bash
!python fast-reid/tools/train_net.py \
    --config-file custom_configs/bagtricks_R50-ibn.yml \
    --eval-only \
    MODEL.WEIGHTS /content/REID_TRAINING/logs/market1501/colab_training/model_final.pth
```

---

## 🎯 Quick Copy-Paste Template

Create a new Colab notebook and paste this:

```python
# ============================================================================
# REID TRAINING ON COLAB - COMPLETE SETUP
# ============================================================================

# 1. Check GPU
!nvidia-smi

# 2. Mount Drive
from google.colab import drive
drive.mount('/content/drive')

# 3. Clone/Copy Your REID_TRAINING Repository (it already includes fast-reid)
import os
import shutil
import subprocess

# Option A: Clone your REID_TRAINING repo from GitHub
try:
    subprocess.run(['git', 'clone', 'https://github.com/YOUR_USERNAME/REID_TRAINING.git'], check=True)
    os.chdir('/content/REID_TRAINING')
    print(f"✓ Cloned REID_TRAINING from GitHub")
except:
    print("✗ Clone failed - trying Drive instead...")
    # Option B: Copy from Drive (if uploaded there)
    try:
        shutil.copytree('/content/drive/My Drive/REID_TRAINING', '/content/REID_TRAINING', dirs_exist_ok=True)
        os.chdir('/content/REID_TRAINING')
        print(f"✓ Copied REID_TRAINING from Drive")
    except:
        print("✗ Both failed - check GitHub URL or upload to Drive")

# Verify fast-reid is included
if os.path.exists('fast-reid'):
    print(f"✓ fast-reid folder found (part of REID_TRAINING)")
else:
    print(f"✗ fast-reid not found - your REID_TRAINING repo should include it")

# 4. Install PyTorch
!pip install --upgrade pip
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 5. Install FastReID Dependencies
!pip install opencv-python faiss-cpu yacs termcolor tabulate cloudpickle tqdm wheel scikit-learn tensorboard

# Install requirements from docs folder
import subprocess
if os.path.exists('fast-reid/docs/requirements.txt'):
    subprocess.run(['pip', 'install', '-r', 'fast-reid/docs/requirements.txt'], check=True)
else:
    print("⚠ requirements.txt not found - main deps already installed")

# 6. Install FastReID Package
import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "-e", "fast-reid"])

# 7. Set Paths
os.environ['PYTHONPATH'] = '/content/REID_TRAINING/fast-reid'
os.environ['FASTREID_DATASETS'] = '/content/REID_TRAINING/fast-reid/datasets'

# 8. Copy Dataset from Drive (if available)
drive_dataset = '/content/drive/My Drive/Market-1501-v15.09.15'
local_dataset = '/content/REID_TRAINING/fast-reid/datasets/Market-1501-v15.09.15'
if os.path.exists(drive_dataset):
    shutil.copytree(drive_dataset, local_dataset, dirs_exist_ok=True)
    print("✓ Dataset copied from Drive")

# 9. Start Training
!python train_research_grade.py \
    --config-file custom_configs/bagtricks_R50-ibn.yml \
    --run-name colab_run

# 10. Save to Drive (RUN BEFORE DISCONNECTING!)
shutil.copytree('/content/REID_TRAINING/logs',
                 '/content/drive/My Drive/REID_Results',
                 dirs_exist_ok=True)
print("✓ Training complete! Results saved to Google Drive")
```

---

## 📚 Additional Resources

- [FastReID GitHub](https://github.com/JDAI-CV/fast-reid)
- [Google Colab Documentation](https://colab.research.google.com)
- [Market-1501 Dataset](https://www.lipreading.com/Market-1501/)
- [PyTorch CUDA Guide](https://pytorch.org/)

---

## ❓ Need Help?

Check the main [README.md](README.md) or [RESEARCHER_GUIDE.txt](RESEARCHER_GUIDE.txt) for more details.

Good luck! 🚀
