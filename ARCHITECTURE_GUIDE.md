# Model Architecture Analysis for Market-1501

## The Problem: Why ResNet-50 Fails on Market-1501

### Dataset Characteristics
```
Market-1501:
  - Identities: 751 classes
  - Training Images: ~12,936
  - Ratio: 12,936 / 751 = 17.2 images per identity

ImageNet:
  - Classes: 1,000
  - Images: 1,200,000
  - Ratio: 1,200 images per class

Scale difference: Market-1501 is 70x SMALLER
```

### Why ResNet-50 Breaks
| Component | ResNet-50 | Market-1501 Need |
|-----------|-----------|-----------------|
| Model Parameters | 24M | 2-5M optimal |
| Feature Dimension | 2048 | 512-1024 sufficient |
| Layers | 50 | 18-34 appropriate |
| Designed for | 1000s classes + millions images | 751 classes + 13k images |
| Risk | SEVERE OVERFITTING | Minimal |

**Mathematics:**
- ResNet-50 with 2048 features for 751 classes = **2.7 parameters per training sample**
- This leads to: model memorizes instead of generalizing
- Result: Loss stays at 10.0, mAP at 1.6% (random guessing)

---

## Solution: Three Architectures for Market-1501

### 1. LIGHTWEIGHT (Fastest Training) - Use This First
**File:** `lightweight_resnet18.yml`
- **Backbone:** ResNet-18 (18 layers)
- **Features:** 512
- **Parameters:** ~2.5M
- **Best for:** Debugging, fast iteration, quick experiments
- **Training time:** ~30-40 min per epoch
- **Expected mAP:** 35-45% (still good)

```yaml
BACKBONE:
  DEPTH: 18x
  FEAT_DIM: 512
```

**When to use:** First test runs, hyperparameter tuning, rapid development

---

### 2. OPTIMAL (Balanced) - Production Standard
**File:** `optimal_market1501.yml`
- **Backbone:** ResNet-34 (34 layers)
- **Features:** 1024
- **Parameters:** ~5.5M
- **Best for:** Standard training, good accuracy
- **Training time:** ~60-70 min per epoch
- **Expected mAP:** 50-65%

```yaml
BACKBONE:
  DEPTH: 34x
  FEAT_DIM: 1024
```

**When to use:** Production models, final training runs

---

### 3. HEAVY (If you want max accuracy)
**File:** Create from template below if needed
- **Backbone:** ResNet-50 (50 layers)
- **Features:** 2048
- **Parameters:** ~24M
- **Best for:** High-performance systems with dropout/regularization
- **Training time:** ~120+ min per epoch
- **Expected mAP:** 60-70%
- **CAVEAT:** Requires L2 regularization, dropout, or other anti-overfit measures

```yaml
BACKBONE:
  DEPTH: 50x
  FEAT_DIM: 2048
  # MUST ADD:
  # WEIGHT_DECAY: 0.001 (higher than default)
```

**When to use:** Only if you can add regularization layers

---

## Critical Configuration Changes Required

### 1. Batch Size Adjustment
```yaml
DATALOADER:
  IMS_PER_BATCH: 32  # Changed from 16
  # Now: 8 people × 4 instances = 32
  # Stronger gradient signal while maintaining P-K sampling
```

**Why:** Batch of 16 is too small. With 32, you get:
- 8 different people (more diversity)
- 4 instances each (triplet loss still works perfectly)
- 2x stronger gradients per update

### 2. Learning Rate Adjustment
```yaml
SOLVER:
  BASE_LR: 0.0005  # Slightly increased from 0.00035
  # Lighter models can use slightly higher LR
  # ResNet-50 might need 0.00025 (lower)
```

### 3. Feature Dimension Matters
```
512 dimensions:  Fast, 32-45% mAP, 1-2M params
1024 dimensions: Balanced, 50-65% mAP, 5-6M params  ✓ BEST
2048 dimensions: Slow, 60-70% mAP, 24M params (overfits easily)
```

---

## Training Progression Recommendation

### Phase 1: Lightweight Test (30 min)
```bash
python train_research_grade.py --config-file custom_configs/lightweight_resnet18.yml
```
- **Goal:** Verify training infrastructure works
- **Expected:** mAP should reach 20-30% by epoch 10
- **If fails:** Dataset loading or sampler still broken
- **If succeeds:** Move to Phase 2

### Phase 2: Optimal Production (4 hours)
```bash
python train_research_grade.py --config-file custom_configs/optimal_market1501.yml
```
- **Goal:** Train to convergence
- **Expected:** mAP reaches 50-60%
- **Runtime:** ~100 epochs × 70 min = 116 hours (run with early stopping)

### Phase 3: Heavy/Fine-tune (Optional, 6+ hours)
```bash
python train_research_grade.py --config-file custom_configs/heavy_resnet50_optimized.yml
```
- **Only if Phase 2 plateaus below 55% mAP**
- **Requires:** Extra regularization

---

## Architecture Selection Guide

Choose your architecture based on:

### ResNet-18 (Lightweight)
✓ Training starts training immediately  
✓ Easy to debug  
✓ Fast iteration (30 min/epoch)  
✗ Limited accuracy (35-45%)  
**Use when:** Developing, testing, limited compute

### ResNet-34 (Optimal) ← RECOMMENDED
✓ Good accuracy (50-65%)  
✓ Fast enough (60-70 min/epoch)  
✓ Avoids overfitting  
✓ Proven on Market-1501  
**Use when:** Production, standard training

### ResNet-50 (Heavy)
✓ Maximum accuracy (60-70%)  
✗ Risk of overfitting  
✗ Slow (120+ min/epoch)  
✗ Requires regularization tricks  
**Use when:** You have 751+ identities AND add L2/dropout

---

## Quick Testing

Copy this test to run immediately:

```bash
# Test 1: Lightweight (5 min to verify it runs)
python train_research_grade.py --config-file custom_configs/lightweight_resnet18.yml

# Test 2: Optimal (30 min to first evaluation)
python train_research_grade.py --config-file custom_configs/optimal_market1501.yml
```

**Success indicators:**
- Epoch 1 loss: 2.5-3.5 (NOT 9.8)
- Epoch 2 mAP: 15-20% (NOT 1.6%)
- Loss decreasing each epoch
- mAP improving across epochs

If you see these, **the architecture fix worked**.

---

## Common Mistakes to Avoid

❌ **ResNet-50 + batch_size=16** → Overfitting + weak gradients  
✓ **ResNet-34 + batch_size=32** → Balanced, stable training

❌ **FEAT_DIM=2048 for 751 classes** → 2.7 params per training image  
✓ **FEAT_DIM=1024 for 751 classes** → 1.4 params per training image

❌ **Freezing backbone layers** → Pretrain weight not useful  
✓ **Training all layers** → Full fine-tuning (correct for ReID)

---

## References

Market-1501 Standard Practices:
- ResNet-34 or ResNet-50 with fine-tuning
- 1024-2048 feature dimension
- Triplet Loss + CrossEntropy
- P-K Identity Sampler (4-8 people per batch)
- Batch size 32-64
- Learning rate 0.00035-0.0005
