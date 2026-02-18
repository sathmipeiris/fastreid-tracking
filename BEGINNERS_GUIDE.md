# FastReID Training - Beginner's Guide

## What is FastReID?

**FastReID** = A deep learning framework to identify the same person in different camera views
- Used for security/surveillance: "Find this person in all cameras"
- Learns to recognize people by their appearance
- Outputs a "signature" (embedding) for each person

**Simple example:**
```
Input:  Photo of person at Camera 1
Output: "This is person #5"

Input:  Photo of person at Camera 2
Output: "This is person #5" (same person!)
```

---

## What is Training?

**Training** = Teaching the model to recognize people

**Analogy:** If a child sees 1000 pictures of dogs, they learn to recognize dogs.

```
Training Process:
1. Show model 1000s of person images (dataset)
2. Model makes guesses ("This looks like person A")
3. We tell it right/wrong answer
4. Model learns from mistakes
5. Repeat thousands of times
6. Model gets better at recognizing people
```

---

## What are Epochs?

**1 Epoch** = Going through the entire dataset once

```
Example: Dataset has 1000 images

Epoch 1: Show model images 1-1000 (learns something)
Epoch 2: Show model images 1-1000 again (learns better)
Epoch 3: Show model images 1-1000 again (learns even better)
...
Epoch 120: Show model images 1-1000 (final training)

Total: 120,000 images shown to model
```

**More epochs = Better accuracy (but takes longer)**

---

## Your Repo Structure Explained

```
REID_TRAINING/
│
├── fast-reid/                          ← The FastReID framework (DO NOT MODIFY)
│   ├── fastreid/                       ← Core code (model, training engine)
│   ├── configs/                        ← Standard settings
│   ├── datasets/                       ← Market-1501 images go here
│   └── tools/
│
├── custom_configs/                     ← YOUR CUSTOM SETTINGS
│   ├── bagtricks_R50-ibn.yml          ← Main config (what we use)
│   ├── test_quick_sanity_check.yml    ← Quick test (1 epoch)
│   └── plateau_solutions/              ← 5 different training strategies
│       ├── solution_1_higher_lr.yml    ← Faster learning
│       ├── solution_2_cosine_annealing.yml
│       ├── solution_3_heavy_triplet.yml ← Better accuracy
│       ├── solution_4_aggressive_lr_drop.yml
│       └── solution_5_smaller_batch_higher_lr.yml
│
├── train_research_grade.py             ← Main training script (WHAT YOU RUN)
├── colab_train.py                      ← Simple Colab wrapper (NEW - USE THIS)
│
└── logs/                               ← Results saved here after training
    └── market1501/
        └── your_run_name/
            ├── model_final.pth         ← Trained model (save this!)
            └── metrics/                ← Training graphs
```

---

## Key Files Explained

### 1. **colab_train.py** (THE ONE YOU RUN - NEW!)
```
What it does:
- Handles all setup automatically
- Downloads dataset from Drive
- Installs libraries
- Runs training
- Saves results

How to use:
!python colab_train.py              # Test with 1 epoch
!python colab_train.py --full       # Full training (90 epochs)
```

### 2. **train_research_grade.py** (Advanced)
```
What it does:
- The actual training engine
- Uses settings from config files
- Saves models and metrics

This is what colab_train.py calls behind the scenes
```

### 3. **custom_configs/bagtricks_R50-ibn.yml** (Settings)
```yaml
# What it means:
MODEL:
  BACKBONE:
    WITH_IBN: True           ← Use special ResNet model
  LOSSES:
    CE:
      SCALE: 0.5             ← Loss weight (don't worry)
    TRI:
      SCALE: 1.5             ← Another loss weight

SOLVER:
  MAX_EPOCH: 90              ← Train for 90 epochs
  IMS_PER_BATCH: 16          ← Show 16 images at a time
  BASE_LR: 0.0005            ← Learning rate (speed of learning)

TEST:
  EVAL_PERIOD: 10            ← Check accuracy every 10 epochs
```

### 4. **plateau_solutions/** (Alternative settings)
```
Problem: Training gets stuck (no improvement)

Solution 1: Higher Learning Rate
  - Train faster
  - Risk: Overshoots, unstable
  
Solution 2: Cosine Annealing
  - Smooth learning curve
  - Stable, predictable
  
Solution 3: Heavy Triplet Loss
  - Better feature learning
  - Longer training time
  - ✓ Usually best accuracy
  
Solution 4: Aggressive LR Drop
  - Fast initial learning
  - Rapid drop later
  
Solution 5: Smaller Batch + High LR
  - For limited GPU memory
  - More chaotic training
```

---

## The Training Workflow (Step-by-Step)

```
1. SETUP
   ├─ Mount Google Drive
   ├─ Copy REID_TRAINING repo
   └─ Install libraries
   
2. PREPARE DATA
   ├─ Download Market-1501 dataset
   └─ Put in fast-reid/datasets/
   
3. TRAIN MODEL
   ├─ Run: !python colab_train.py
   ├─ Model learns from 1000s of images
   ├─ Every 10 epochs: check accuracy
   └─ Save best model
   
4. RESULTS
   ├─ Model saved to: logs/market1501/your_run_name/model_final.pth
   ├─ Metrics saved (charts, accuracy)
   └─ Download to your computer
```

---

## What's the Dataset?

**Market-1501** = A dataset with:
- 1,501 different people
- ~33,000 images
- Each person photographed by 6 different cameras
- Goal: Recognize same person across cameras

```
Person #1:
├─ Camera A: 5 images
├─ Camera B: 4 images
├─ Camera C: 6 images
└─ ... (total: ~22 images of same person)

Person #2: 20 images
Person #3: 18 images
...
Person #1501: 22 images
```

---

## Quick Reference: What to Do

### To Run a Test (5 minutes, 1 epoch)
```python
!python colab_train.py
```
✓ Quick check if everything works

### To Run Full Training (3 hours, 90 epochs)
```python
!python colab_train.py --full
```
✓ Gets decent accuracy (~70% rank-1)

### To Try Different Strategy
```python
# Solution 1 (Recommended)
!python colab_train.py --full --run-name sol1

# Solution 3 (Best quality, longer)
!python colab_train.py --config custom_configs/plateau_solutions/solution_3_heavy_triplet.yml --epochs 120 --eval-period 10 --run-name sol3
```

---

## Understanding Training Output

```
When training runs, you'll see:
[Epoch  10/90] loss: 2.34, rank-1: 45.2%, mAP: 32.1%
[Epoch  20/90] loss: 1.89, rank-1: 62.3%, mAP: 45.2%
[Epoch  30/90] loss: 1.45, rank-1: 72.1%, mAP: 54.3%
...
[Epoch  90/90] loss: 0.87, rank-1: 82.5%, mAP: 68.9%

What it means:
- loss: Should go DOWN ↓ (good = model learning)
- rank-1: Should go UP ↑ (% correct first guess)
- mAP: Should go UP ↑ (overall accuracy)

Target: rank-1 > 80%, mAP > 60%
```

---

## Common Issues & Fixes

### "ModuleNotFoundError: fastreid.data.datasets"
```
Problem: Wrong folder structure
Fix: Make sure you copied REID_TRAINING (with fast-reid inside)
     NOT fastreid-tracking alone
```

### GPU not available
```
Problem: CPU only training (very slow)
Solution: Use Google Colab (free GPU)
          or train with fewer epochs
```

### Out of memory (OOM)
```
Problem: Batch size too large
Fix: Edit config SOLVER.IMS_PER_BATCH: 8 (reduce from 16)
```

### Training stuck at 50% accuracy
```
Problem: Learning rate wrong or model plateau
Fix: Try plateau_solutions/solution_3_heavy_triplet.yml
```

---

## Final Summary

**You have 3 options:**

### Option 1: EASIEST (Recommended!)
```python
!python colab_train.py --full
# Automates everything, good results
```

### Option 2: Full Control
```python
!python train_research_grade.py \
    --config-file custom_configs/plateau_solutions/solution_1_higher_lr.yml \
    --run-name my_experiment \
    SOLVER.MAX_EPOCHS 90 \
    TEST.EVAL_PERIOD 10
```

### Option 3: Explore
```python
# Try all 5 solutions, see which works best
# Takes longer but gives best results
```

---

**Still confused?** Ask me specific questions and I'll explain!
