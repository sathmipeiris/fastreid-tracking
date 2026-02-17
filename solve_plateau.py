#!/usr/bin/env python
# encoding: utf-8
"""
Solution for Loss Plateau Problem
Addresses high CE loss and stagnant triplet loss
"""

import os
from pathlib import Path


def create_plateau_solver_configs():
    """
    Create optimized configs to break the plateau based on observed issues:
    - CE Loss stuck at 6.71-6.73 (too high)
    - Triplet Loss at 2.7-2.9 (stagnant)
    - Overall loss: 9.5-9.6 (barely moving)
    
    Solutions:
    1. Increase learning rate (may be too conservative)
    2. Reweight losses (reduce CE, increase Triplet)
    3. Add label smoothing to CE loss
    4. Ensure hard triplet mining is active
    5. Try cosine annealing scheduler instead of fixed steps
    """
    
    base_template = """_BASE_: ../fast-reid/configs/Base-bagtricks.yml

MODEL:
  BACKBONE:
    WITH_IBN: True
  HEADS:
    WITH_BNNECK: True
  LOSSES:
    CE:
      SCALE: {ce_weight}
    TRI:
      SCALE: {tri_weight}
      HARD_MINING: True

DATASETS:
  NAMES: ("Market1501",)
  TESTS: ("Market1501",)

DATALOADER:
  NUM_WORKERS: 0

OUTPUT_DIR: logs/market1501/plateau_solver

SOLVER:
  MAX_EPOCH: {max_epoch}
  IMS_PER_BATCH: {batch_size}
  BASE_LR: {learning_rate}
  AMP:
    ENABLED: False
  CHECKPOINT_PERIOD: 120
  EARLY_STOP_PATIENCE: 15
  WARMUP_ITERS: 1000
  SCHED: {scheduler}
  {scheduler_params}

TEST:
  EVAL_PERIOD: 1
"""

    configs = [
        {
            'name': 'solution_1_higher_lr',
            'description': 'Problem: Learning rate too low. Solution: 2x learning rate (0.0007 vs 0.00035)',
            'params': {
                'ce_weight': '0.5',  # Reduce CE weight
                'tri_weight': '1.5',  # Increase Triplet weight
                'max_epoch': '90',
                'batch_size': '16',
                'learning_rate': '0.0007',  # 2x increase
                'scheduler': 'MultiStepLR',
                'scheduler_params': 'STEPS: [30, 60, 80]',
            }
        },
        {
            'name': 'solution_2_cosine_annealing',
            'description': 'Problem: Fixed LR schedule causes plateau. Solution: Cosine annealing',
            'params': {
                'ce_weight': '0.5',
                'tri_weight': '1.5',
                'max_epoch': '90',
                'batch_size': '16',
                'learning_rate': '0.0005',
                'scheduler': 'CosineAnnealingLR',
                'scheduler_params': 'ETA_MIN: 0.00001',
            }
        },
        {
            'name': 'solution_3_heavy_triplet',
            'description': 'Problem: CE loss dominates. Solution: Strong triplet loss weighting',
            'params': {
                'ce_weight': '0.3',  # Very low
                'tri_weight': '2.0',  # Very high
                'max_epoch': '120',
                'batch_size': '16',
                'learning_rate': '0.0005',
                'scheduler': 'MultiStepLR',
                'scheduler_params': 'STEPS: [40, 80, 110]',
            }
        },
        {
            'name': 'solution_4_aggressive_lr_drop',
            'description': 'Problem: Learning rate doesn\'t drop enough. Solution: Aggressive drops at epochs 20, 50',
            'params': {
                'ce_weight': '0.5',
                'tri_weight': '1.5',
                'max_epoch': '80',
                'batch_size': '16',
                'learning_rate': '0.001',  # Higher initial
                'scheduler': 'MultiStepLR',
                'scheduler_params': 'STEPS: [20, 50, 70]\n  GAMMA: 0.1',
            }
        },
        {
            'name': 'solution_5_smaller_batch_higher_lr',
            'description': 'Problem: Batch too large smooths out hard negatives. Solution: Smaller batch + higher LR',
            'params': {
                'ce_weight': '0.5',
                'tri_weight': '1.5',
                'max_epoch': '100',
                'batch_size': '8',  # Smaller batch
                'learning_rate': '0.001',  # Much higher
                'scheduler': 'MultiStepLR',
                'scheduler_params': 'STEPS: [30, 70, 90]',
            }
        },
    ]
    
    output_dir = Path('custom_configs/plateau_solutions')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print("CREATING PLATEAU SOLUTION CONFIGS")
    print(f"{'='*70}\n")
    
    for config in configs:
        filename = f"{config['name']}.yml"
        filepath = output_dir / filename
        
        # Format params into template
        content = base_template.format(**config['params'])
        
        with open(filepath, 'w') as f:
            f.write(content)
        
        print(f"✓ {filename}")
        print(f"  {config['description']}\n")
    
    return configs


def create_comparison_script():
    """Create script to run all plateau solutions."""
    
    script = """#!/bin/bash
# Run all plateau solutions and compare results

echo "=================================================="
echo "PLATEAU SOLUTION COMPARISON"
echo "=================================================="
echo ""

SOLUTIONS=(
    "solution_1_higher_lr"
    "solution_2_cosine_annealing"
    "solution_3_heavy_triplet"
    "solution_4_aggressive_lr_drop"
    "solution_5_smaller_batch_higher_lr"
)

for SOLUTION in "${SOLUTIONS[@]}"; do
    echo "Starting: $SOLUTION"
    python train_research_grade.py \\
        --config-file custom_configs/plateau_solutions/${SOLUTION}.yml \\
        --run-name $SOLUTION
    echo "Completed: $SOLUTION\\n"
done

echo "=================================================="
echo "All solutions tested. Comparing results..."
python find_best_model.py --compare-only --output-dir logs/plateau_solutions
"""
    
    # Windows batch version
    batch_script = """@echo off
REM Run all plateau solutions and compare results

setlocal enabledelayedexpansion

echo ==================================================
echo PLATEAU SOLUTION COMPARISON
echo ==================================================
echo.

set "SOLUTIONS=solution_1_higher_lr solution_2_cosine_annealing solution_3_heavy_triplet solution_4_aggressive_lr_drop solution_5_smaller_batch_higher_lr"

for %%S in (%SOLUTIONS%) do (
    echo Starting: %%S
    python train_research_grade.py ^
        --config-file custom_configs/plateau_solutions/%%S.yml ^
        --run-name %%S
    echo Completed: %%S
    echo.
)

echo ==================================================
echo All solutions tested. Comparing results...
python find_best_model.py --compare-only --output-dir logs/plateau_solutions

pause
"""
    
    # Save batch version (Windows)
    with open('run_plateau_solutions.bat', 'w') as f:
        f.write(batch_script)
    
    print(f"✓ Created: run_plateau_solutions.bat")


def print_recommendations():
    """Print detailed recommendations."""
    
    print(f"""
{'='*70}
PLATEAU ANALYSIS & RECOMMENDED SOLUTIONS
{'='*70}

PROBLEM ANALYSIS:
─────────────────────────────────────────────────────────────────────
Your model shows:
  • CE Loss: 6.71-6.73 (very high and stagnant)
  • Triplet Loss: 2.7-2.9 (improving but slow)
  • Total Loss: 9.5-9.6 (barely changing)
  • No progress from epoch 10-29

ROOT CAUSES:
─────────────────────────────────────────────────────────────────────
1. Learning rate likely TOO LOW
   → Model not making progress despite good convergence
   → Especially CE loss needs higher gradient updates

2. Loss imbalance (CE >> Triplet)
   → CE loss at 6.71 vs Triplet at 2.8
   → CE loss dominates gradient flow
   → Solution: Reduce CE weight, increase Triplet weight

3. Fixed learning rate schedule may not fit data
   → Linear MultiStepLR might not be optimal
   → Try Cosine Annealing for smoother descent

SOLUTIONS PROVIDED:
─────────────────────────────────────────────────────────────────────
5 automatically generated configs addressing each issue:

✓ solution_1_higher_lr
  → 2x learning rate (0.0007)
  → Balanced loss weights
  → TRY THIS FIRST if you want single solution

✓ solution_2_cosine_annealing
  → Cosine annealing scheduler
  → Better for continuous improvement
  → Best for finding optimal plateaus

✓ solution_3_heavy_triplet
  → Minimal CE weight (0.3)
  → Maximum Triplet weight (2.0)
  → Best for metric learning focus

✓ solution_4_aggressive_lr_drop
  → Higher initial LR (0.001)
  → Aggressive drops at epochs 20, 50
  → Best if data requires coarse-to-fine learning

✓ solution_5_smaller_batch_higher_lr
  → Batch size 8 (vs 16)
  → Higher learning rate (0.001)
  → Best for hard sample mining

HOW TO RUN:
─────────────────────────────────────────────────────────────────────
Option 1: Run ALL solutions (recommended for research)
  > run_plateau_solutions.bat
  (This will test all 5 and show which is best)

Option 2: Run single solution
  > python train_research_grade.py \\
      --config-file custom_configs/plateau_solutions/solution_1_higher_lr.yml \\
      --run-name solution_1_higher_lr

Option 3: Use hyperparameter search
  > python find_best_model.py --base-config custom_configs/plateau_solutions/solution_1_higher_lr.yml

EXPECTED OUTCOMES:
─────────────────────────────────────────────────────────────────────
After fixing plateau, expect:
  ✓ CE loss dropping from 6.7 → 5.0-5.5
  ✓ Triplet loss improving to 2.0-2.5
  ✓ Total loss > 8.5
  ✓ mAP improving continuously
  ✓ Better validation metrics

DEPLOYMENT:
─────────────────────────────────────────────────────────────────────
After finding best solution:
  1. Check STOPPING_REASON.txt in the winning run's directory
  2. Use best_model.pth from that run
  3. Copy to deployment: logs/final_model/best_model.pth
  4. Load in enrollment_tracker.py

{'='*70}
""")


if __name__ == '__main__':
    import sys
    
    print("""
╔════════════════════════════════════════════════════════════════════╗
║                   PLATEAU SOLVER CONFIGURATION                     ║
║   Automatic solutions for loss plateau & high CE loss issues       ║
╚════════════════════════════════════════════════════════════════════╝
""")
    
    # Create configs
    configs = create_plateau_solver_configs()
    
    # Create comparison script
    create_comparison_script()
    
    # Print recommendations
    print_recommendations()
    
    print(f"\n{'='*70}")
    print("NEXT STEP: Run")
    print("  > run_plateau_solutions.bat")
    print("To test all 5 solutions and automatically select the best one.")
    print(f"{'='*70}\n")
