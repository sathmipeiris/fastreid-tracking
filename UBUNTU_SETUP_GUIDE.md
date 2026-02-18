# FastReID Training on Ubuntu - Complete Guide

## Step 1: Install Python & Git

```bash
# Update system
sudo apt update
sudo apt upgrade -y

# Install Python 3.9+
sudo apt install python3 python3-pip git -y

# Verify installation
python3 --version
pip3 --version
```

---

## Step 2: Clone the Repository

```bash
# Clone from GitHub
git clone https://github.com/sathmipeiris/fastreid-tracking.git
cd fastreid-tracking

# Switch to main branch (if needed)
git checkout main
```

---

## Step 3: Create Python Virtual Environment

```bash
# Create virtual environment
python3 -m venv reid_env

# Activate it
source reid_env/bin/activate

# You should see (reid_env) at the start of your terminal
```

---

## Step 4: Install Dependencies

```bash
# Make sure you're in the repo directory
cd fastreid-tracking

# Install required packages
pip install --upgrade pip

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 opencv-python faiss-cpu yacs scikit-learn tensorboard termcolor tqdm Pillow numpy scipy matplotlib tabulate pyyaml six setuptools wheel gdown

# Install fast-reid in development mode
pip install -e fast-reid/
```

**Note:** If you have NVIDIA GPU, use this for PyTorch instead:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

---

## Step 5: Download Dataset

### Option A: Download Market-1501 (Recommended)

```bash
# Create dataset folder
mkdir -p fast-reid/datasets

# Download Market-1501 (you'll need an account at https://zheng-lab.cecs.anu.edu.au/Project/project_reid.html)
# Once downloaded, extract it:
cd fast-reid/datasets
unzip Market-1501-v15.09.15.zip
cd ../..
```

### Option B: Use Included Dataset (if already there)

```bash
# If dataset is already in the folder, nothing to do
ls -la fast-reid/datasets/
```

---

## Step 6: Verify Setup

```bash
# Test imports
python3 -c "import torch; print('PyTorch:', torch.__version__)"
python3 -c "import fastreid; print('FastReID OK')"
python3 -c "import cv2; print('OpenCV OK')"

# Check dataset
ls -la fast-reid/datasets/Market-1501-v15.09.15/
```

Expected output:
```
PyTorch: 2.0.1+cu118
FastReID OK
OpenCV OK
bounding_box_test/
bounding_box_train/
query/
```

---

## Step 7: Run Training

### Quick Test (4 epochs - 5 minutes - Validation Only)

```bash
# Activate virtual environment first
source reid_env/bin/activate

# Run with test config (does NOT save trained model, just validates setup)
python3 train_research_grade.py \
    --config-file custom_configs/test_quick_sanity_check.yml
```

### Full Training (90 epochs - 6-8 hours - Saves Trained Model)

```bash
source reid_env/bin/activate

python3 train_research_grade.py \
    --config-file custom_configs/bagtricks_R50-ibn.yml \
    --run-name production_run
```

### With Custom Parameters

```bash
source reid_env/bin/activate

python3 train_research_grade.py \
    --config-file custom_configs/bagtricks_R50-ibn.yml \
    --run-name my_experiment \
    SOLVER.MAX_EPOCHS 120 \
    SOLVER.BASE_LR 0.0005 \
    TEST.EVAL_PERIOD 10
```

---

## Step 8: Monitor Training

```bash
# In another terminal, view TensorBoard logs
source reid_env/bin/activate

tensorboard --logdir logs/market1501/
# Visit: http://localhost:6006
```

---

## Step 9: Get Results

After training completes:

```bash
# Check saved model
ls -la logs/market1501/production_run/

# Files created:
# - model_final.pth         ← Your trained model
# - events.out.tfevents.*   ← TensorBoard logs
# - config.yaml             ← Settings used
```

---

## Step 10: Extract & Visualize Metrics (THESIS QUALITY)

Generate high-quality graphs for your thesis evaluation:

```bash
# Activate virtual environment
source reid_env/bin/activate

# Generate all evaluation graphs (300 DPI - thesis quality)
python3 evaluate_and_export_metrics.py \
    --log-dir logs/market1501/production_run \
    --output-dir logs/market1501/production_run/metrics_graphs
```

This generates 9 graphs:
- ✅ Training Loss Curve
- ✅ mAP vs Epoch (Mean Average Precision)
- ✅ Rank-1 Accuracy vs Epoch  
- ✅ Cosine Similarity Histogram (matched vs non-matched pairs)
- ✅ ROC Curve (identification performance)
- ✅ Confusion Matrix
- ✅ FPS vs Time Graph (processing performance)
- ✅ Temporal Smoothing Effect (raw vs smoothed scores)
- ✅ ID Stability Graph (tracking robustness)

All saved as PNG at 300 DPI (publication quality).

```bash
# View generated graphs
ls -la logs/market1501/production_run/metrics_graphs/
```

---

## Critical Metrics for Thesis Evaluation

**ReID Performance:**
- `mAP` - Mean Average Precision (how well it ranks similar people)
- `Rank-1` - Top-1 accuracy (most critical metric)
- `Rank-5` - Top-5 accuracy
- Loss curve - Shows model learning over time

**Identification Quality:**
- Cosine similarity histogram - Shows decision boundary between same/different persons
- ROC curve - Shows TPR vs FPR at different thresholds
- Confusion matrix - TP, FP, TN, FN breakdown

**System Performance:**
- FPS graph - Real-time processing speed
- Temporal smoothing - Denoising effectiveness
- ID stability - Tracking consistency over time

---

## Complete Workflow Script

Copy this as a single script `run_training.sh`:

```bash
#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}FastReID Ubuntu Training Setup${NC}"

# Step 1: Create venv
echo -e "${GREEN}1. Creating virtual environment...${NC}"
python3 -m venv reid_env
source reid_env/bin/activate

# Step 2: Install dependencies
echo -e "${GREEN}2. Installing dependencies...${NC}"
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 opencv-python faiss-cpu yacs scikit-learn tensorboard termcolor tqdm Pillow numpy scipy matplotlib tabulate pyyaml six setuptools wheel gdown
pip install -e fast-reid/

# Step 3: Verify
echo -e "${GREEN}3. Verifying installation...${NC}"
python3 -c "import torch; print('PyTorch OK')"
python3 -c "import fastreid; print('FastReID OK')"

# Step 4: Check dataset
echo -e "${GREEN}4. Checking dataset...${NC}"
if [ -d "fast-reid/datasets/Market-1501-v15.09.15" ]; then
    echo "Dataset found!"
else
    echo "⚠️  Dataset not found. Download from: https://zheng-lab.cecs.anu.edu.au/Project/project_reid.html"
fi

# Step 5: Run training
echo -e "${GREEN}5. Starting training...${NC}"
python3 train_research_grade.py \
    --config-file custom_configs/bagtricks_R50-ibn.yml \
    --run-name ubuntu_run

echo -e "${GREEN}✓ Training complete!${NC}"
echo "Results in: logs/market1501/ubuntu_run/"
```

Run it:
```bash
chmod +x run_training.sh
./run_training.sh
```

---

## Troubleshooting

### "ModuleNotFoundError: No module named 'fastreid'"
```bash
# Are you in the virtual environment?
source reid_env/bin/activate

# Reinstall fast-reid
pip install -e fast-reid/
```

### CUDA out of memory
```bash
# Reduce batch size in config
SOLVER.IMS_PER_BATCH: 8  # change from 16
```

### No GPU found
```bash
# Check CUDA availability
python3 -c "import torch; print(torch.cuda.is_available())"

# If False, check NVIDIA drivers
nvidia-smi
```

### Slow training (CPU only)
```bash
# Install NVIDIA CUDA
# See: https://docs.nvidia.com/cuda/cuda-installation-guide-linux/

# Then reinstall PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

---

## Summary

```bash
# TL;DR - 4 commands:
git clone https://github.com/sathmipeiris/fastreid-tracking.git
cd fastreid-tracking
python3 -m venv reid_env
source reid_env/bin/activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 opencv-python faiss-cpu yacs scikit-learn tensorboard termcolor tqdm Pillow numpy scipy matplotlib tabulate pyyaml six setuptools wheel gdown && pip install -e fast-reid/
python3 train_research_grade.py --config-file custom_configs/bagtricks_R50-ibn.yml
```

That's it! Training starts automatically.
