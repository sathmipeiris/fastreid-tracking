# Quick Reference: Early Stopping Training

## ⚡ Quick Start (60 seconds)

```bash
# 1. Run training
run_training_with_early_stopping.bat

# 2. Wait for completion (~8 hours)

# 3. Analyze results
python analyze_training.py logs/market1501/bagtricks_R50-ibn/validation_history.json --plot
```

---

## 📊 What Gets Monitored Every Epoch

| Metric | Purpose | Target |
|--------|---------|--------|
| **mAP** | Ranking accuracy | 0.40-0.55+ |
| **Top-1** | Rank-1 accuracy | 0.55-0.70+ |
| **Train Loss** | Training convergence | <2.5 |
| **Overfitting Flag** | Validation degradation | No flags |

---

## 🎯 Early Stopping Decision Tree

```
Does mAP improve from last epoch?
│
├─ YES → Save as best model, reset patience counter
│
└─ NO  → Increment patience counter
         │
         ├─ patience < 10 → Continue training
         │
         └─ patience ≥ 10 → STOP (no improvement for 10 epochs)
```

---

## 📁 Output Files Location

```
logs/market1501/bagtricks_R50-ibn/
├── best_model.pth          ← Use this for deployment
├── model_final.pth         ← Final checkpoint (if completed all epochs)
├── validation_history.json ← Analysis data
├── log.txt                 ← Training logs
└── config.yaml             ← Configuration used
```

---

## 🔴 Red Flags (Something Wrong?)

| Issue | Check | Fix |
|-------|-------|-----|
| No validation output | GPU memory full? | Reduce batch size |
| mAP not improving | Poor initialization? | Run longer, check config |
| Early stopping at epoch 5 | PATIENCE too low? | Increase to 15 |
| mAP fluctuates wildly | Learning rate too high? | Reduce BASE_LR |

---

## 📈 Expected Training Curve

```
Epoch 1:  mAP=0.10  ↑↑↑ Rapid improvement
Epoch 10: mAP=0.35  ↑↑  Good progress
Epoch 20: mAP=0.45  ↑   Slower improvement
Epoch 30: mAP=0.48  →   Plateau begins (EARLY STOP LIKELY HERE)
Epoch 40: mAP=0.48  →   No improvement for 10 epochs
Epoch 40: STOPPED! ✓   Training ends, best=0.48@epoch 30
```

---

## 🛠️ Common Adjustments

### If early stopping too aggressive (stops at epoch 20):
```yaml
SOLVER:
  EARLY_STOP_PATIENCE: 15  # Was 10, now more patient
```

### If early stopping never triggers (runs all 60 epochs):
```yaml
SOLVER:
  MAX_EPOCH: 120  # Let it run longer
```

### If validation takes too long:
```yaml
TEST:
  EVAL_PERIOD: 5    # Validate every 5 epochs instead of 1
  IMS_PER_BATCH: 32 # Reduce batch size for faster evaluation
```

---

## 💾 Model Selection

**For deployment use:**
```python
# Load the best model (highest validation mAP)
best_checkpoint = torch.load('logs/market1501/bagtricks_R50-ibn/best_model.pth')

# NOT the final model
# final_checkpoint = torch.load('logs/market1501/bagtricks_R50-ibn/model_final.pth')
```

---

## 📊 Real Overfitting Examples

### ✅ Normal (Training continues):
```
Epoch 30: train_loss=1.8, mAP=0.45 (LR drop ×0.1)
Epoch 31: train_loss=1.9, mAP=0.46 ← Spike is normal
Epoch 32: train_loss=1.7, mAP=0.47 ← Recovery
```

### ❌ Real Overfitting (Early stop would trigger):
```
Epoch 30: train_loss=1.0, mAP=0.50
Epoch 31: train_loss=0.9, mAP=0.49 ← Diverging
Epoch 32: train_loss=0.8, mAP=0.48 ← FLAG
```

---

## 📝 Console Output Meanings

```
✓ New best mAP! Saving...
→ Model improved, patience counter reset

No improvement. Epochs: 1/10
→ One epoch without improvement (out of 10 patience)

⚠ Overfitting detected!
→ Validation degrading while training improves

EARLY STOPPING: mAP hasn't improved for 10 epochs
→ Training automatically stopped
```

---

## 🎓 Key Differences from Before

| Before | Now |
|--------|-----|
| Validation every 30 epochs | Validation every epoch ✓ |
| No early stopping | Early stopping at 10 epochs ✓ |
| No overfitting detection | Automatic detection ✓ |
| Lost best model | Separate best_model.pth ✓ |
| No analysis tools | analyze_training.py ✓ |
| Can't detect plateaus | Plateaus detected ✓ |

---

## ✅ Success Checklist

Before starting training, verify:
- [ ] `train_with_early_stopping.py` exists
- [ ] `analyze_training.py` exists
- [ ] Config has `EARLY_STOP_PATIENCE: 10`
- [ ] Config has `TEST.EVAL_PERIOD: 1`
- [ ] Market1501 dataset exists in `fast-reid/datasets/`
- [ ] Virtual environment activated
- [ ] CUDA is available (`nvidia-smi` works)

After training completes:
- [ ] `best_model.pth` created (>200 MB)
- [ ] `validation_history.json` created (<50 KB)
- [ ] mAP improved from epoch 1
- [ ] Early stopping triggered OR max epochs reached
- [ ] Analysis runs without errors

---

## 🚀 One-Line Reference

```bash
# Do this to train
run_training_with_early_stopping.bat

# Then do this to analyze
python analyze_training.py logs/market1501/bagtricks_R50-ibn/validation_history.json --plot

# Then use this model
best_model.pth
```

---

## 📞 Troubleshooting

**Training seems stuck at "Validation"?**
- First epoch validation takes 2-5 min (full ReID eval)
- Be patient for first epoch

**mAP very low (<0.1)?**
- Check FASTREID_DATASETS env variable
- Check dataset path in config
- Ensure model initialized correctly

**Can't find validation_history.json?**
- Check if training actually completed
- Check output directory name in config
- Look in `logs/market1501/bagtricks_R50-ibn/`

**Want to adjust early stopping?**
- Edit `custom_configs/bagtricks_R50-ibn.yml`
- Change `EARLY_STOP_PATIENCE: 10` (or other value)
- Run training again
