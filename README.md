# Target-Specific Identity Enrollment & Tracking

This project implements a Person Re-Identification (ReID) system using FastReID.

## Installation

1. Run `setup_env.bat` to create the environment and install dependencies.
2. The script will also clone `fast-reid` (if not present) and install it.

## Dataset

Download Market1501 and place it in `fast-reid/datasets/Market-1501-v15.09.15/`.

## Training

```bash
conda activate fastreid
python fast-reid/tools/train_net.py --config-file custom_configs/bagtricks_R50-ibn.yml OUTPUT_DIR logs/market_r50_ibn
```

## Running the Tracker

```bash
python enrollment_tracker.py --source 0 --opts MODEL.WEIGHTS logs/market_r50_ibn/model_final.pth
```
