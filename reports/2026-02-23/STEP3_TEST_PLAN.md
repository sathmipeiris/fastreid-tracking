# STEP 3 & 4: BASELINE TEST EXECUTION PLAN
**Date**: February 23, 2026  
**Config**: Official FastReID Market1501 Baseline  
**Target**: Verify baseline works before debugging custom configs

---

## 📋 TEST SPECIFICATION

### What We Are Testing
```bash
python fast-reid/tools/train_net.py \
  --config-file custom_configs/baseline_official_R50.yml \
  OUTPUT_DIR logs/baseline_test_clean
```

### Configuration Details
- **Base**: `fast-reid/configs/Base-bagtricks.yml` (unchanged official)
- **Dataset**: Market1501 (751 identities, 12,936 training images)
- **Model**: ResNet-50 backbone, 2048-dim features
- **Loss**: CrossEntropyLoss + TripletLoss (official)
- **Sampler**: NaiveIdentitySampler (official)
- **Batch Size**: 64 (official)
- **Learning Rate**: 0.00035 (official)
- **Max Epochs**: 120 (official)
- **Warmup**: 2000 iterations (official)

---

## 🎯 SUCCESS CRITERIA

### By Epoch 5 (expected first evaluation)
```
✅ PASS if:
   - Loss < 5.0
   - Rank-1 > 40%
   - mAP > 30%
   
❌ FAIL if:
   - Loss > 10.0
   - Rank-1 < 10%
   - mAP < 5%
```

### By Epoch 20
```
✅ PASS if:
   - Rank-1 > 75%
   - mAP > 65%
```

---

## 📊 METRICS EXPLANATION

| Metric | Meaning | Healthy Value |
|--------|---------|---|
| **Loss** | Total training loss (CE + Triplet) | < 3.0 by epoch 20 |
| **Rank-1** | Is ground-truth person #1 nearest match? | > 75% |
| **Rank-5** | Is GT person in top-5 ranks? | > 85% |
| **mAP** | Average Precision across all queries | > 65% |

**What each means**:
- **High Loss + Low Rank-1**: Model hasn't learned anything (random)
- **Low Loss + High Rank-1**: Model learned embeddings perfectly

---

## 🚀 EXECUTION STEPS

1. Activate venv
2. Navigate to workspace root
3. Run training command with official config
4. Monitor outputs for 5-10 epochs
5. Check metrics vs success criteria
6. Document results in test report

---

## ⏱️ EXPECTED DURATION

- **Per Epoch**: ~2-3 minutes (GPU) or ~10-15 minutes (CPU)
- **First 5 Epochs**: ~10-15 minutes (GPU) or ~50-75 minutes (CPU)
- **Full Test (10 epochs)**: ~20-30 minutes (GPU) or ~100-150 minutes (CPU)

---

## 📝 WHAT TO WATCH FOR

### Green Flag Indicators
- ✅ Loss decreasing each epoch
- ✅ Rank-1 increasing each epoch  
- ✅ No GPU/memory errors
- ✅ Checkpoints saved successfully

### Red Flag Indicators
- ❌ Loss stuck or increasing
- ❌ Rank-1 near 0% (model didn't learn)
- ❌ Out of memory errors
- ❌ NaN or Inf values

---

**TEST PLAN READY**: Prepared for execution February 23, 2026
