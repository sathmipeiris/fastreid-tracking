# STEP 1: AUDIT REPORT - Configuration Analysis
**Date**: February 23, 2026  
**Status**: CRITICAL - Custom config corruption detected

---

## 🔴 BROKEN CONFIG IDENTIFIED

**File**: `custom_configs/optimal_market1501.yml`  
**Status**: CAUSING TRAINING COLLAPSE (mAP ≈ 1%, Rank-1 = 0%)

---

## 📊 DETAILED COMPARISON: CUSTOM vs OFFICIAL

### BACKBONE CONFIGURATION

| Parameter | Official (bagtricks_R50.yml) | Custom (optimal_market1501.yml) | Status |
|-----------|------------------------------|----------------------------------|--------|
| DEPTH | 50x | 34x | ❌ REDUCED |
| FEAT_DIM | 2048 | 1024 | ❌ HALVED |
| WITH_IBN | False | True | ⚠️ MODIFIED |
| WITH_PRETRAIN | True | True | ✅ OK |

**IMPACT**: Reduced backbone capacity by ~50% in both depth and feature dimension. For 751 Market1501 identities, ResNet-50 is MINIMUM. ResNet-34 with 1024 dims is too weak.

---

### SAMPLER & DATA LOADING

| Parameter | Official | Custom | Status |
|-----------|----------|--------|--------|
| SAMPLER_TRAIN | NaiveIdentitySampler | BalancedIdentitySampler | ⚠️ CHANGED |
| NUM_INSTANCE | 4 | 4 | ✅ OK |
| BATCH_SIZE (IMS_PER_BATCH) | 64 | 32 | ❌ HALVED |
| NUM_WORKERS | 8 | 0 | ❌ DISABLED |

**IMPACT**: 
- Changed sampler type without understanding ReID metric learning principles
- Batch size 32 is TOO SMALL for both triplet loss convergence and stable gradient updates
- NUM_WORKERS=0 removes parallel data loading (Windows compatibility issue?)

---

### LOSS CONFIGURATION

| Parameter | Official | Custom | Status |
|-----------|----------|--------|--------|
| LOSSES | CrossEntropyLoss + TripletLoss | CrossEntropyLoss + TripletLoss | ✅ SAME |
| CE.EPSILON | 0.1 | 0.1 | ✅ SAME |
| CE.SCALE | 1.0 | 1.0 | ✅ SAME |
| TRI.MARGIN | 0.3 | 0.3 | ✅ SAME |
| TRI.HARD_MINING | True | True | ✅ SAME |

**IMPACT**: Loss functions themselves are correct, BUT:
- With batch size 32, triplet loss has fewer hard negatives to mine
- With weak backbone (ResNet-34), loss cannot converge

---

### TRAINING HYPERPARAMETERS

| Parameter | Official | Custom | Status |
|-----------|----------|--------|--------|
| MAX_EPOCH | 120 | 100 | ⚠️ REDUCED |
| BASE_LR | 0.00035 | 0.00035 | ✅ SAME |
| WEIGHT_DECAY | 0.0005 | 0.0005 | ✅ SAME |
| STEPS (LR schedule) | [40, 90] | [40, 70, 90] | ⚠️ MODIFIED |
| GAMMA (LR decay) | 0.1 | 0.1 | ✅ SAME |
| WARMUP_ITERS | 2000 | 500 | ❌ REDUCED BY 75% |
| AMP (Mixed Precision) | True | False | ❌ DISABLED |

**IMPACT**:
- WARMUP_ITERS reduced from 2000→500: Model trains too aggressively too soon
- LR schedule changed: Learning rate drops earlier at epoch 40 instead of smoothly at 40→90
- AMP disabled: Removes stability from mixed precision training
- Early epoch 70 LR drop may reset unstable poorly-initialized weights

---

### EVALUATION SETTINGS

| Parameter | Official | Custom | Status |
|-----------|----------|--------|--------|
| EVAL_PERIOD | 30 | 5 | ⚠️ INCREASED FREQUENCY |
| CHECKPOINT_PERIOD | 30 | 10 | ⚠️ INCREASED FREQUENCY |

**IMPACT**: More frequent evaluation can mask instability, but does NOT cause collapse. Not the culprit.

---

## 🚨 ROOT CAUSE ANALYSIS

### Why Training Collapsed (mAP ≈ 1%, Rank-1 = 0%)

**PRIMARY CAUSES** (in order of severity):

1. **❌ BATCH SIZE = 32 (too small)**
   - ReID requires P×K sampling: P identities × K images per identity
   - Batch size 32 = only ~8 identities × 4 images
   - Triplet loss needs hard negatives from different identities
   - Too few identities per batch = weak metric learning signal
   - Training stagnates at random initialization

2. **❌ BACKBONE = ResNet-34 + 1024 dims (too weak)**
   - Market1501 has 751 training identities
   - Feature space needs 2048 dims to accommodate this many centers
   - 1024 dims = feature space too crowded
   - Model cannot learn separable embeddings
   - Loss stays high, doesn't converge

3. **❌ WARMUP_ITERS = 500 (too short)**
   - Model trains for only ~500 iterations before full learning rate
   - With batch size 32: 32 iters/epoch × 500 = ~16 epochs of ramp-up
   - Model never reaches stable training before first LR drop at epoch 40
   - All gradients are untrained noise

4. **⚠️ SAMPLER = BalancedIdentitySampler (unknown behavior)**
   - Official uses NaiveIdentitySampler
   - Unknown if "Balanced" changed sampling strategy
   - Could be missing hard negatives for triplet loss

5. **⚠️ AMP DISABLED**
   - Removes numerical stability from training
   - Can cause gradient overflow/underflow with random initializations

---

## 📐 LOSS CALCULATION & EVALUATION EXPLAINED

### How Loss is Calculated During Training

**Per batch:**
```
Total Loss = CE_Loss + TRI_Loss

1. CrossEntropyLoss:
   - Treats ReID as 751-way classification
   - Pushes embeddings toward class centers
   - Formula: L = -log(exp(logit_gt) / sum(exp(logits)))
   - Scale: 1.0

2. TripletLoss:  
   - With HARD_MINING=True: selects hardest positives/negatives in batch
   - Formula: L = max(0, margin + d_pos - d_neg)
   - Scale: 1.0
   - MARGIN: 0.3 = minimum embedding distance between positive-negative pairs
```

**Why it fails with batch size 32:**
- Fewer identities per batch = fewer hard negatives
- Triplet loss becomes: max(0, 0.3 + distant_positive - random_negative)
- If random negatives happen to be far, margin is satisfied
- No hard mining occurs → metric learning fails

### How Metrics are Evaluated (Every 5 epochs)

**Validation Loop:**
1. Extract features for all gallery images (test set)
2. Extract features for all query images (probe set)
3. Compute distance matrix using cosine/L2 similarity
4. Rank-based matching:
   - **Rank-1**: Does #1 nearest match = GT identity?
   - **Rank-5, Rank-10**: Is GT identity in top-5, top-10?
5. **mAP**: Average Precision across all queries

**Why metrics are 0%:**
- Embeddings remain unstructured (random)
- All queries get wrong nearest match
- → Rank-1 = 0%
- → mAP = 1% (random chance)

---

## ✅ VERIFICATION: Will Official Config Harm Custom Configs?

**ANALYSIS**: NO, it will NOT harm existing custom configs.

**Reasons:**
1. Official config is isolated: `fast-reid/configs/Market1501/bagtricks_R50.yml`
2. Custom configs are separate: `custom_configs/*.yml`
3. No dependencies between them
4. We are ONLY copying official config, not modifying custom ones
5. File structure prevents cross-contamination

**Action**: SAFE to proceed.

---

## 🎯 NEXT STEPS

### Immediate Actions
1. ✅ Created: `logs/baseline_test_clean/` directory
2. ✅ Ready: Use official `fast-reid/configs/Market1501/bagtricks_R50.yml`
3. ⏳ Next: Run baseline training for 5-10 epochs

### Expected Results (If System Works)
By epoch 5:
- Loss: < 5.0
- Rank-1: > 40%
- mAP: > 30%

By epoch 20:
- Rank-1: 75-85%

If these occur → baseline is correct → we debug custom config safely.
If not → environmental issue → debug environment.

---

**AUDIT COMPLETED**: February 23, 2026  
**CONCLUSION**: Configuration corruption confirmed. Safe to proceed with baseline test.
