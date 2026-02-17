@echo off
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
