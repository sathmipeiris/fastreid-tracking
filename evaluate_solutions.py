#!/usr/bin/env python
# encoding: utf-8
"""
Evaluate all plateau solution models and compare results.
Loads trained model_final.pth from each solution and tests on Market1501.
"""

import sys
import os
import json
import logging
from pathlib import Path
from datetime import datetime
from collections import OrderedDict

# Setup path
if 'fast-reid' not in sys.path:
    sys.path.insert(0, 'fast-reid')

if sys.platform == "win32":
    os.system('color')

import torch
from fastreid.config import get_cfg
from fastreid.engine import DefaultTrainer, default_argument_parser, default_setup
from fastreid.utils import comm
from fastreid.utils.checkpoint import Checkpointer

# Color codes
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def evaluate_model(config_file, model_path, solution_name):
    """
    Evaluate a single model.
    
    Args:
        config_file: Path to config file
        model_path: Path to model_final.pth
        solution_name: Name of the solution for display
        
    Returns:
        dict with mAP, top-1, and other metrics
    """
    logger = logging.getLogger(__name__)
    
    # Load config
    cfg = get_cfg()
    cfg.merge_from_file(config_file)
    cfg.defrost()
    cfg.MODEL.BACKBONE.PRETRAIN = False
    # Disable multiprocessing to avoid paging file issues on Windows
    cfg.DATALOADER.NUM_WORKERS = 0
    cfg.freeze()
    
    # Setup
    default_setup(cfg, None)
    
    # Build model
    model = DefaultTrainer.build_model(cfg)
    
    # Load checkpoint
    if not os.path.exists(model_path):
        logger.error(f"Model not found: {model_path}")
        return None
    
    try:
        Checkpointer(model).load(model_path)
        logger.info(f"✓ Loaded model: {model_path}")
    except Exception as e:
        logger.error(f"Failed to load model {model_path}: {e}")
        return None
    
    # Evaluate
    try:
        results = DefaultTrainer.test(cfg, model)
        return results
    except Exception as e:
        logger.error(f"Evaluation failed for {solution_name}: {e}")
        return None


def main():
    """Evaluate all solutions and create comparison."""
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # Solutions to evaluate
    solutions = [
        ('solution_1_higher_lr', 'custom_configs/plateau_solutions/solution_1_higher_lr.yml'),
        ('solution_2_cosine_annealing', 'custom_configs/plateau_solutions/solution_2_cosine_annealing.yml'),
        ('solution_3_heavy_triplet', 'custom_configs/plateau_solutions/solution_3_heavy_triplet.yml'),
        ('solution_4_aggressive_lr_drop', 'custom_configs/plateau_solutions/solution_4_aggressive_lr_drop.yml'),
        ('solution_5_smaller_batch_higher_lr', 'custom_configs/plateau_solutions/solution_5_smaller_batch_higher_lr.yml'),
    ]
    
    results_dict = {}
    
    # Header
    logger.info("")
    logger.info(f"{Colors.BOLD}{Colors.CYAN}┏{'━'*70}┓{Colors.ENDC}")
    logger.info(f"{Colors.BOLD}{Colors.CYAN}┃{' '*15}PLATEAU SOLUTION EVALUATION{' '*29}┃{Colors.ENDC}")
    logger.info(f"{Colors.BOLD}{Colors.CYAN}┗{'━'*70}┛{Colors.ENDC}")
    logger.info("")
    
    # Evaluate each solution
    for solution_name, config_file in solutions:
        model_path = f"logs/market1501/plateau_solver/{solution_name}/model_final.pth"
        
        logger.info(f"{Colors.BOLD}{Colors.BLUE}Evaluating: {solution_name}{Colors.ENDC}")
        
        if not os.path.exists(model_path):
            logger.warning(f"  ✗ Model not found: {model_path}")
            logger.warning(f"    (Train this solution first)")
            results_dict[solution_name] = None
            logger.info("")
            continue
        
        results = evaluate_model(config_file, model_path, solution_name)
        
        if results:
            results_dict[solution_name] = results
            
            # Display results
            mAP = float(results.get('mAP', 0.0))
            top1 = float(results.get('top-1', 0.0))
            top5 = float(results.get('top-5', 0.0))
            
            map_color = Colors.GREEN if mAP > 0.3 else Colors.YELLOW if mAP > 0.1 else Colors.RED
            
            logger.info(f"  {Colors.BOLD}Results:{Colors.ENDC}")
            logger.info(f"    mAP:    {map_color}{mAP:.4f}{Colors.ENDC}")
            logger.info(f"    top-1:  {Colors.CYAN}{top1:.4f}{Colors.ENDC}")
            if top5 > 0:
                logger.info(f"    top-5:  {Colors.CYAN}{top5:.4f}{Colors.ENDC}")
        else:
            results_dict[solution_name] = None
        
        logger.info("")
    
    # Create comparison table
    logger.info(f"{Colors.BOLD}{Colors.CYAN}┏{'━'*70}┓{Colors.ENDC}")
    logger.info(f"{Colors.BOLD}{Colors.CYAN}┃{' '*20}COMPARISON TABLE{' '*34}┃{Colors.ENDC}")
    logger.info(f"{Colors.BOLD}{Colors.CYAN}┗{'━'*70}┛{Colors.ENDC}")
    logger.info("")
    
    # Filter results that exist
    valid_results = {k: v for k, v in results_dict.items() if v is not None}
    
    if not valid_results:
        logger.warning("No models evaluated successfully. Train solutions first.")
        return
    
    # Sort by mAP
    sorted_results = sorted(
        valid_results.items(),
        key=lambda x: float(x[1].get('mAP', 0.0)),
        reverse=True
    )
    
    # Print table
    logger.info(f"{Colors.BOLD}{'Rank':<6} {'Solution':<35} {'mAP':<10} {'Top-1':<10}{Colors.ENDC}")
    logger.info(f"{Colors.CYAN}{'─'*6} {'─'*35} {'─'*10} {'─'*10}{Colors.ENDC}")
    
    for rank, (solution_name, results) in enumerate(sorted_results, 1):
        mAP = float(results.get('mAP', 0.0))
        top1 = float(results.get('top-1', 0.0))
        
        # Color code based on rank
        if rank == 1:
            rank_color = Colors.GREEN
            medal = "🏆"
        elif rank == 2:
            rank_color = Colors.YELLOW
            medal = "🥈"
        elif rank == 3:
            rank_color = Colors.YELLOW
            medal = "🥉"
        else:
            rank_color = Colors.CYAN
            medal = "  "
        
        map_color = Colors.GREEN if mAP > 0.3 else Colors.YELLOW
        
        logger.info(
            f"{rank_color}{rank}. {medal}{Colors.ENDC} {solution_name:<32} "
            f"{map_color}{mAP:.4f}{Colors.ENDC}     {top1:.4f}"
        )
    
    logger.info("")
    
    # Best solution
    best_solution, best_results = sorted_results[0]
    best_mAP = float(best_results.get('mAP', 0.0))
    
    logger.info(f"{Colors.BOLD}{Colors.GREEN}🏆 BEST SOLUTION:{Colors.ENDC}")
    logger.info(f"  {Colors.GREEN}{best_solution}{Colors.ENDC}")
    logger.info(f"  mAP: {Colors.GREEN}{best_mAP:.4f}{Colors.ENDC}")
    logger.info("")
    
    # Save results to file
    output_file = 'EVALUATION_RESULTS.json'
    
    evaluation_summary = {
        'timestamp': datetime.now().isoformat(),
        'best_solution': best_solution,
        'best_mAP': float(best_mAP),
        'all_results': {}
    }
    
    for solution_name, results in results_dict.items():
        if results:
            evaluation_summary['all_results'][solution_name] = {
                'mAP': float(results.get('mAP', 0.0)),
                'top-1': float(results.get('top-1', 0.0)),
                'top-5': float(results.get('top-5', 0.0)),
                'model_path': f"logs/market1501/plateau_solver/{solution_name}/model_final.pth"
            }
        else:
            evaluation_summary['all_results'][solution_name] = {
                'status': 'not_evaluated',
                'reason': 'model_not_found_or_evaluation_failed'
            }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(evaluation_summary, f, indent=2)
    
    logger.info(f"{Colors.BOLD}{Colors.GREEN}✓ Results saved to:{Colors.ENDC} {Colors.YELLOW}{output_file}{Colors.ENDC}")
    logger.info(f"{Colors.BOLD}{Colors.CYAN}┗{'━'*70}┛{Colors.ENDC}")
    logger.info("")


if __name__ == "__main__":
    main()
