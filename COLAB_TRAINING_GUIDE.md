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

### Cell 2: Copy Your REID_TRAINING Repository from Drive (Recommended)
```python
import os
import shutil

os.chdir('/content')

# Copy from Drive (you upload REID_TRAINING.zip to Drive first)
try:
    print("Copying REID_TRAINING from Drive...")
    shutil.copytree('/content/drive/MyDrive/REID_TRAINING', '/content/REID_TRAINING', dirs_exist_ok=True)
    print("✓ Repository copied from Drive")
    os.chdir('/content/REID_TRAINING')
    print(f"✓ Working directory: {os.getcwd()}")
    print(f"✓ fast-reid exists: {os.path.exists('fast-reid')}")
except Exception as e:
    print(f"✗ Copy failed: {e}")
    print("Make sure you uploaded REID_TRAINING folder to Drive root")
```

**How to prepare:**
1. Zip your local `c:\Users\sathmi\Desktop\REID_TRAINING` folder
2. Upload **REID_TRAINING.zip** to Google Drive root
3. Run this cell in Colab

### Cell 3: Install PyTorch for Colab
```python
!pip install --upgrade pip
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Cell 4: Install FastReID Dependencies
```python
import os
import subprocess

# Verify we're in the right directory
os.chdir('/content/REID_TRAINING')
print(f"Current directory: {os.getcwd()}")

# Check that fast-reid exists (it should - it's in your repo)
if not os.path.exists('fast-reid'):
    print("✗ ERROR: fast-reid folder not found!")
    print("Make sure you uploaded the correct REID_TRAINING folder from Drive")
else:
    print("✓ fast-reid folder found")

# Install common dependencies
!pip install opencv-python faiss-cpu yacs termcolor tabulate cloudpickle tqdm wheel scikit-learn tensorboard

# Install from requirements.txt (skip if any fail, main deps already installed)
if os.path.exists('fast-reid/docs/requirements.txt'):
    print("✓ Installing from fast-reid/docs/requirements.txt...")
    try:
        subprocess.run(['pip', 'install', '-r', 'fast-reid/docs/requirements.txt'], check=True)
    except:
        print("⚠ Some requirements failed (but main deps are installed - proceeding anyway)")
else:
    print("⚠ requirements.txt not found - main deps already installed")
```

### Cell 5: Setup FastReID Module Path
```python
import sys
import os

# Set up paths
os.chdir('/content/REID_TRAINING')

# Add fast-reid to Python path (simpler & more reliable than pip install -e)
if '/content/REID_TRAINING/fast-reid' not in sys.path:
    sys.path.insert(0, '/content/REID_TRAINING/fast-reid')
    print("✓ FastReID path added to sys.path")

# Verify fastreid can be imported
try:
    import fastreid
    print(f"✓ FastReID imported successfully")
    print(f"  Location: {fastreid.__file__}")
except ImportError as e:
    print(f"⚠ Import issue: {e}")
    print("  But setup.py-based install isn't needed - path is set")
    print("  Training should still work with dependencies installed")

# Double-check key dependencies are available
required = ['torch', 'cv2', 'yacs', 'faiss', 'sklearn']
missing = []
for pkg in required:
    try:
        __import__(pkg)
    except ImportError:
        missing.append(pkg)

if missing:
    print(f"⚠ Missing packages: {missing}")
    print("  Install with: !pip install opencv-python faiss-cpu scikit-learn")
else:
    print(f"✓ All core dependencies available")
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
# If dataset is in your Google Drive as a ZIP file
import shutil
import zipfile
import os

# These are the paths to check in your Drive
drive_dataset_dir = '/content/drive/My Drive/Market-1501-v15.09.15'
drive_dataset_zip = '/content/drive/My Drive/Market-1501-v15.09.15.zip'

local_dataset_dir = '/content/REID_TRAINING/fast-reid/datasets/Market-1501-v15.09.15'
local_dataset_dir_parent = '/content/REID_TRAINING/fast-reid/datasets'

os.makedirs(local_dataset_dir_parent, exist_ok=True)

# Try to find dataset: unzipped folder first, then zip file
if os.path.exists(drive_dataset_dir):
    print("✓ Found unzipped dataset in Drive, copying...")
    shutil.copytree(drive_dataset_dir, local_dataset_dir, dirs_exist_ok=True)
    print(f"✓ Dataset copied from Drive: {local_dataset_dir}")

elif os.path.exists(drive_dataset_zip):
    print(f"✓ Found zip file in Drive, downloading and extracting...")
    # Copy zip to Colab first (faster than direct extraction)
    local_zip = f'{local_dataset_dir_parent}/Market-1501.zip'
    print(f"  Copying {drive_dataset_zip} to {local_zip}...")
    shutil.copy(drive_dataset_zip, local_zip)
    
    # Extract
    print(f"  Extracting {local_zip}...")
    with zipfile.ZipFile(local_zip, 'r') as zip_ref:
        zip_ref.extractall(local_dataset_dir_parent)
    
    # Clean up zip
    os.remove(local_zip)
    print(f"✓ Dataset extracted to: {local_dataset_dir}")
else:
    print("✗ Dataset not found in Drive!")
    print("  Expected either:")
    print(f"    - Folder: /My Drive/Market-1501-v15.09.15")
    print(f"    - ZIP: /My Drive/Market-1501-v15.09.15.zip")
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
import os
import subprocess

# CRITICAL: Make sure you're in the right directory
os.chdir('/content/REID_TRAINING')
print(f"Working dir: {os.getcwd()}")

# Quick test with 1 epoch + evaluation to verify everything works
result = subprocess.run([
    'python', 'train_research_grade.py',
    '--config-file', 'custom_configs/bagtricks_R50-ibn.yml',
    '--run-name', 'colab_test_1epoch',
    'SOLVER.MAX_EPOCHS', '1',
    'TEST.EVAL_PERIOD', '1'
], cwd='/content/REID_TRAINING')

if result.returncode == 0:
    print("\n✓ Test complete! Check metrics in next cell with TensorBoard")
else:
    print(f"\n✗ Training failed with error code {result.returncode}")
```

**Advanced: Use Plateau Solutions for Better Convergence**

After test passes, try one of these optimized configs:
```python
import os
os.chdir('/content/REID_TRAINING')

# Solution 1: Higher Learning Rate (faster convergence)
!python train_research_grade.py \
    --config-file custom_configs/plateau_solutions/solution_1_higher_lr.yml \
    --run-name solution_1_higher_lr \
    TEST.EVAL_PERIOD 10

# Solution 2: Cosine Annealing (smooth learning decay)
# !python train_research_grade.py \
#     --config-file custom_configs/plateau_solutions/solution_2_cosine_annealing.yml \
#     --run-name solution_2_cosine \
#     TEST.EVAL_PERIOD 10

# Solution 3: Heavy Triplet Loss (better feature separation)
# !python train_research_grade.py \
#     --config-file custom_configs/plateau_solutions/solution_3_heavy_triplet.yml \
#     --run-name solution_3_heavy_triplet \
#     TEST.EVAL_PERIOD 10

# Solution 4: Aggressive LR Drop (quick learning rate reduction)
# !python train_research_grade.py \
#     --config-file custom_configs/plateau_solutions/solution_4_aggressive_lr_drop.yml \
#     --run-name solution_4_aggressive \
#     TEST.EVAL_PERIOD 10

# Solution 5: Smaller Batch + Higher LR (more gradient updates)
# !python train_research_grade.py \
#     --config-file custom_configs/plateau_solutions/solution_5_smaller_batch_higher_lr.yml \
#     --run-name solution_5_batch8 \
#     TEST.EVAL_PERIOD 10
```

### Cell 8: Monitor Training & View Metrics
```python
# FIRST: Load TensorBoard extension in Colab
%load_ext tensorboard

# THEN: Launch TensorBoard to visualize real-time metrics
%tensorboard --logdir /content/REID_TRAINING/logs

# TensorBoard will show:
# - Training Loss (Cross-Entropy + Triplet Loss)
# - Validation Rank-1 Accuracy
# - Validation mAP (Mean Average Precision)
# - Learning Rate Schedule
# - False Positives / Negative distances
# - Confusion matrices per epoch
```

**What to look for in TensorBoard:**
- **Loss curve:** Should decrease smoothly (not spike)
- **Rank-1 accuracy:** Should increase towards convergence (80-90%+)
- **mAP:** Higher is better (50-80% is good)
- **Learning rate:** Should follow your SCHED setting (MultiStepLR, CosineAnnealingLR, etc.)

### Cell 9: Compare Training Results Across Solutions
```python
import json
from pathlib import Path
import pandas as pd

logs_dir = Path('/content/REID_TRAINING/logs/market1501')

# Find all training runs
runs = {}
for run_dir in logs_dir.glob('*/'):
    log_file = run_dir / 'log.txt'
    if log_file.exists():
        runs[run_dir.name] = run_dir

# Create comparison table
results = []
for run_name, run_path in runs.items():
    log_file = run_path / 'log.txt'
    try:
        # Extract final metrics (simplified parsing)
        with open(log_file, 'r') as f:
            lines = f.readlines()
            final_line = lines[-1] if lines else ""
            results.append({
                'Run': run_name,
                'Path': str(run_path),
                'Logs': str(log_file)
            })
    except:
        pass

if results:
    df = pd.DataFrame(results)
    print("\n" + "="*80)
    print("TRAINING RUNS COMPARISON")
    print("="*80)
    print(df.to_string(index=False))
    print("="*80)
else:
    print("No completed runs found yet. Training is in progress or just started.")
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

## 🎯 Plateau Solutions Comparison

When training plateaus (no improvement), try one of these 5 optimized configurations:

| Solution | Strategy | Key Params | When to Use |
|----------|----------|-----------|------------|
| **Solution 1: Higher LR** | Increase learning rate to escape plateau | BASE_LR: 0.0007, MultiStepLR [30,60,80] | Quick convergence, high-quality data |
| **Solution 2: Cosine Annealing** | Smooth learning rate decay | BASE_LR: 0.0005, CosineAnnealingLR | Fine-tuning, smooth training curves |
| **Solution 3: Heavy Triplet** | Emphasize triplet loss (2.0 vs 1.5) | TRI.SCALE: 2.0, MAX_EPOCHS: 120 | Better feature separation, needs more epochs |
| **Solution 4: Aggressive LR Drop** | Rapid LR reduction early | BASE_LR: 0.001, STEPS: [20,50,70] | Quick optimization, may overfit |
| **Solution 5: Small Batch + High LR** | More gradient updates per epoch | IMS_PER_BATCH: 8, BASE_LR: 0.001 | Limited GPU memory, noisy gradients |

**Recommendation for Colab (Free Tesla T4):**
- Start with **Solution 1** (faster, fewer epochs)
- If stuck, try **Solution 2** (smoother learning)
- For best accuracy, use **Solution 3** (longer training, better features)

---

## ⚙️ Advanced Configuration

### Quick Sanity Check
```bash
!python train_research_grade.py \
    --config-file custom_configs/test_quick_sanity_check.yml \
    --run-name quick_sanity \
    TEST.EVAL_PERIOD 1
```

### Enable Evaluation During Training
```bash
# Add TEST.EVAL_PERIOD to any config to evaluate metrics
# EVAL_PERIOD: 1 = evaluate every epoch
# EVAL_PERIOD: 10 = evaluate every 10 epochs
# EVAL_PERIOD: 0 = disable evaluation (faster training)

!python train_research_grade.py \
    --config-file custom_configs/plateau_solutions/solution_1_higher_lr.yml \
    --run-name with_metrics \
    TEST.EVAL_PERIOD 10  # This enables metrics saving!
```

### Customize Batch Size for Memory
```bash
# Reduce if Out of Memory errors occur
!python train_research_grade.py \
    --config-file custom_configs/bagtricks_R50-ibn.yml \
    --run-name memory_optimized \
    DATALOADER.IMS_PER_BATCH 8 \
    TEST.EVAL_PERIOD 10
```

### Use Different Backbones
```bash
# ResNet50 with IBN (default)
# For lightweight inference, try MobileNet or EfficientNet configs
!python train_research_grade.py \
    --config-file custom_configs/bagtricks_R50-ibn.yml \
    --run-name r50_ibn_baseline
```

---

## 🔧 Troubleshooting

### No Metrics Being Saved (TensorBoard Empty)
```python
# Make sure TEST.EVAL_PERIOD is set to a number > 0
# 0 = disabled, 1-10 = evaluate every N epochs

!python train_research_grade.py \
    --config-file custom_configs/bagtricks_R50-ibn.yml \
    --run-name with_metrics \
    TEST.EVAL_PERIOD 1  # This MUST be > 0!

# Then check logs
!ls -la /content/REID_TRAINING/logs/market1501/with_metrics/
```

### Training Loss Not Decreasing
- Try **Solution 1** (higher learning rate)
- Try **Solution 2** (cosine annealing for smooth learning)
- Check: Is loss NaN or infinite? → Learning rate too high
- Check: Is loss flat? → Learning rate too low (try Solution 1)

### Rank-1 Accuracy Stuck at Low Values
- Try **Solution 3** (heavier triplet loss for better feature learning)
- Increase MAX_EPOCHS (more time to train)
- Check dataset: Are there enough positive pairs per identity?

### Out of Memory (OOM) Error
```python
# Reduce batch size
!python train_research_grade.py \
    --config-file custom_configs/bagtricks_R50-ibn.yml \
    --run-name oom_fix \
    DATALOADER.IMS_PER_BATCH 8 \
    TEST.EVAL_PERIOD 10

# Or try Solution 5 (already optimized for small batch)
!python train_research_grade.py \
    --config-file custom_configs/plateau_solutions/solution_5_smaller_batch_higher_lr.yml \
    --run-name solution_5
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

## 📊 Interpreting Training Metrics

### Key Metrics to Monitor

| Metric | Good Range | What It Means | Action if Low |
|--------|-----------|---------------|---------------|
| **Loss** | Decreases from 5→1 | Training is improving | Check learning rate |
| **Rank-1 Accuracy** | 80-95% | % of queries with correct match at top | Try Solution 3 (triplet loss) |
| **mAP (Mean Avg Precision)** | 50-80% | Overall retrieval quality | Increase epochs or try Solution 3 |
| **Learning Rate** | Following schedule | Should decay over time | Should follow your SCHED [MultiStepLR/CosineAnnealingLR] |

### Real-Time Monitoring with TensorBoard
```python
# During training, in a new cell:
%tensorboard --logdir /content/REID_TRAINING/logs

# Look for:
# - Loss curves (should be smooth, not jagged)
# - Accuracy increasing (not plateauing)
# - Learning rate following your schedule [SCHED]
# - Validation metrics improving (if EVAL_PERIOD > 0)
```

### Which Solution to Try Based on Training Behavior

**Scenario 1: Training loss decreasing but accuracy flat**
→ Try **Solution 3** (Heavy Triplet Loss) - improves feature separation

**Scenario 2: Training loss decreasing too slowly**
→ Try **Solution 1** or **Solution 4** (Higher LR or aggressive drops)

**Scenario 3: Training loss fluctuates wildly**
→ Try **Solution 2** (Cosine Annealing) - smoother learning curve

**Scenario 4: Out of Memory with batch size 16**
→ Use **Solution 5** (batch size 8 with higher LR)

**Scenario 5: No time for long training**
→ Use **Solution 4** (faster convergence with aggressive LR drops)

---

### Download Results
```python
from google.colab import files
import os

# Download best model
model_path = '/content/REID_TRAINING/logs/market1501/*/model_final.pth'
import glob
models = glob.glob(model_path)

if models:
    latest_model = max(models, key=os.path.getctime)
    files.download(latest_model)
    print(f"✓ Downloaded: {latest_model}")
else:
    print("✗ No model found - training may still be in progress")
```

### View Training Metrics (After Training)
```python
# Read and display final metrics
from pathlib import Path
import json

logs_dir = Path('/content/REID_TRAINING/logs/market1501')

for run_dir in logs_dir.glob('*/'):
    metrics_file = run_dir / 'metrics.json'
    if metrics_file.exists():
        with open(metrics_file) as f:
            metrics = json.load(f)
            print(f"\n{'='*60}")
            print(f"Run: {run_dir.name}")
            print(f"{'='*60}")
            for key, val in metrics.items():
                if isinstance(val, float):
                    print(f"{key:.<40} {val:.4f}")
                else:
                    print(f"{key:.<40} {val}")
```

### Compare Multiple Training Runs
```python
# Create comparison table across all runs
import pandas as pd
from pathlib import Path

logs_dir = Path('/content/REID_TRAINING/logs/market1501')
results = []

for run_dir in logs_dir.glob('*/'):
    log_file = run_dir / 'log.txt'
    if log_file.exists():
        results.append({
            'Run Name': run_dir.name,
            'Logs Path': str(run_dir),
            'Has Metrics': (run_dir / 'metrics.json').exists()
        })

if results:
    df = pd.DataFrame(results)
    print(df.to_string(index=False))
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

# 3. Copy Your REID_TRAINING Repository from Drive
import os
import shutil

os.chdir('/content')

print("Copying REID_TRAINING from Drive...")
try:
    shutil.copytree('/content/drive/MyDrive/REID_TRAINING', '/content/REID_TRAINING', dirs_exist_ok=True)
    os.chdir('/content/REID_TRAINING')
    print(f"✓ Copied REID_TRAINING from Google Drive")
    print(f"✓ fast-reid exists: {os.path.exists('fast-reid')}")
except Exception as e:
    print(f"✗ Failed: {e}")
    print("Upload REID_TRAINING.zip to Drive root first")

# 4. Install PyTorch
!pip install --upgrade pip
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 5. Install FastReID Dependencies
!pip install opencv-python faiss-cpu yacs termcolor tabulate cloudpickle tqdm wheel scikit-learn tensorboard

# Install requirements from docs folder (skip if fails - main deps already installed)
import subprocess
if os.path.exists('fast-reid/docs/requirements.txt'):
    try:
        subprocess.run(['pip', 'install', '-r', 'fast-reid/docs/requirements.txt'], timeout=120)
    except:
        print("⚠ Some requirements failed - main deps already installed, proceeding")
else:
    print("⚠ requirements.txt not found - main deps already installed")

# 6. Install FastReID Package
import sys
os.chdir('/content/REID_TRAINING')
if '/content/REID_TRAINING/fast-reid' not in sys.path:
    sys.path.insert(0, '/content/REID_TRAINING/fast-reid')
print("✓ FastReID module path configured")

# 7. Set Paths
os.environ['PYTHONPATH'] = '/content/REID_TRAINING/fast-reid'
os.environ['FASTREID_DATASETS'] = '/content/REID_TRAINING/fast-reid/datasets'

# 8. Copy Dataset from Drive (if available - handles both zip and unzipped)
import zipfile
drive_dataset_dir = '/content/drive/My Drive/Market-1501-v15.09.15'
drive_dataset_zip = '/content/drive/My Drive/Market-1501-v15.09.15.zip'
local_dataset_dir = '/content/REID_TRAINING/fast-reid/datasets/Market-1501-v15.09.15'
local_dataset_parent = '/content/REID_TRAINING/fast-reid/datasets'

os.makedirs(local_dataset_parent, exist_ok=True)

if os.path.exists(drive_dataset_dir):
    shutil.copytree(drive_dataset_dir, local_dataset_dir, dirs_exist_ok=True)
    print("✓ Dataset copied from Drive")
elif os.path.exists(drive_dataset_zip):
    print("Extracting dataset zip...")
    local_zip = f'{local_dataset_parent}/Market-1501.zip'
    shutil.copy(drive_dataset_zip, local_zip)
    with zipfile.ZipFile(local_zip, 'r') as zip_ref:
        zip_ref.extractall(local_dataset_parent)
    os.remove(local_zip)
    print("✓ Dataset extracted from Drive zip")
else:
    print("⚠ Dataset not in Drive - will need to upload or download separately")

# 9. Start Training (test with 1 epoch first, WITH metrics)
import os
import subprocess

os.chdir('/content/REID_TRAINING')
print(f"Working in: {os.getcwd()}")

# Run test
subprocess.run([
    'python', 'train_research_grade.py',
    '--config-file', 'custom_configs/bagtricks_R50-ibn.yml',
    '--run-name', 'test_1epoch',
    'SOLVER.MAX_EPOCHS', '1',
    'TEST.EVAL_PERIOD', '1'
])

print("\n✓ Test run complete! If successful, run full training with plateau solution:")

# After test passes, uncomment ONE of these for full training:
# SOLUTION 1: Higher Learning Rate (RECOMMENDED for Colab)
# os.chdir('/content/REID_TRAINING')
# subprocess.run([
#     'python', 'train_research_grade.py',
#     '--config-file', 'custom_configs/plateau_solutions/solution_1_higher_lr.yml',
#     '--run-name', 'solution_1_v1',
#     'TEST.EVAL_PERIOD', '10'
# ])

# SOLUTION 3: Heavy Triplet (best accuracy, longer)
# os.chdir('/content/REID_TRAINING')
# subprocess.run([
#     'python', 'train_research_grade.py',
#     '--config-file', 'custom_configs/plateau_solutions/solution_3_heavy_triplet.yml',
#     '--run-name', 'solution_3_v1',
#     'TEST.EVAL_PERIOD', '10'
# ])

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
