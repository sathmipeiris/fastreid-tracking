#!/usr/bin/env python
# encoding: utf-8
"""
Researcher-Grade Training System for ReID Model Selection
Automatically tests hyperparameter configurations and finds the best model.
"""

import sys
import os
import json
import logging
from pathlib import Path
from datetime import datetime
from collections import OrderedDict
from tqdm import tqdm

# Add fast-reid to Python path for module resolution
if 'fast-reid' not in sys.path:
    sys.path.insert(0, 'fast-reid')

# Enable ANSI color codes on Windows (required for colored progress bars)
if sys.platform == "win32":
    os.system('color')

import torch

print("Loading training framework...", flush=True)

from fastreid.config import get_cfg
from fastreid.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from fastreid.utils import comm

print("Framework loaded successfully", flush=True)

# Import metric plotter
try:
    from train_metric_plotter import ReIDMetricPlotter, generate_all_plots
    PLOTTER_AVAILABLE = True
except ImportError:
    PLOTTER_AVAILABLE = False
    print("Warning: train_metric_plotter not available - no auto plots will be generated")

# Color codes for terminal output
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


def setup_clean_logging():
    """
    Configure logging for clean ResearchTrainer output.
    Suppresses verbose fastreid iteration logging while ensuring ResearchTrainer output is visible.
    """
    # Suppress fastreid's verbose iteration-level logging ONLY
    logging.getLogger('fastreid.engine.train_loop').setLevel(logging.WARNING)
    logging.getLogger('fastreid.utils.events').setLevel(logging.WARNING)
    logging.getLogger('fastreid.engine.hooks').setLevel(logging.WARNING)
    
    # Create direct console handler for ResearchTrainer output
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    # Simple format that preserves color codes
    formatter = logging.Formatter('%(message)s')
    console_handler.setFormatter(formatter)
    
    # Get the ResearchTrainer logger and configure it directly
    logger = logging.getLogger(__name__)
    logger.handlers.clear()  # Remove any existing handlers
    logger.addHandler(console_handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False  # Don't send logs up to verbose fastreid loggers
    
    return logger


def extract_metric_value(metric_value):
    """
    Extract scalar value from fast-reid metric storage.
    Metrics can be stored as scalars or as (value, count) tuples.
    """
    if metric_value is None:
        return 0.0
    
    # Handle tuple format (value, count) from EventStorage
    if isinstance(metric_value, (tuple, list)):
        return float(metric_value[0]) if metric_value else 0.0
    
    # Handle scalar format
    try:
        return float(metric_value)
    except (ValueError, TypeError):
        return 0.0


class ResearchTrainer(DefaultTrainer):
    """
    Advanced trainer with:
    - Detailed stopping reason logging
    - Model performance tracking
    - Hyperparameter variation support
    - Systematic model comparison
    """
    
    def __init__(self, cfg, run_name="default", hyperparams=None):
        super().__init__(cfg)
        
        # Setup clean logging early
        logger = setup_clean_logging()
        
        self.run_name = run_name
        self.hyperparams = hyperparams or {}
        self.cfg = cfg  # Store config for later access
        
        # Tracking
        self.best_mAP = 0.0
        self.best_epoch = 0
        self.epochs_without_improvement = 0
        self.should_stop = False
        self.stopping_reason = "Unknown"
        
        # History
        self.training_history = {
            'epoch': [],
            'mAP': [],
            'top1': [],
            'train_loss': [],
            'ce_loss': [],
            'triplet_loss': [],
        }
        
        # Paths
        self.run_dir = os.path.join(cfg.OUTPUT_DIR, run_name)
        os.makedirs(self.run_dir, exist_ok=True)
        
        self.stop_reason_file = os.path.join(self.run_dir, 'STOPPING_REASON.txt')
        self.model_info_file = os.path.join(self.run_dir, 'model_info.json')
        self.history_file = os.path.join(self.run_dir, 'training_history.json')
        
        self.patience = cfg.SOLVER.EARLY_STOP_PATIENCE
        self.max_epoch = cfg.SOLVER.MAX_EPOCH
        
        logger = logging.getLogger(__name__)
        
        # Beautiful header
        logger.info("")
        logger.info(f"{Colors.BOLD}{Colors.CYAN}┏{'━'*68}┓{Colors.ENDC}")
        logger.info(f"{Colors.BOLD}{Colors.CYAN}┃{' '*15}ReID RESEARCH-GRADE TRAINER{' '*26}┃{Colors.ENDC}")
        logger.info(f"{Colors.BOLD}{Colors.CYAN}┗{'━'*68}┛{Colors.ENDC}")
        logger.info("")
        logger.info(f"{Colors.BOLD}Run Name:{Colors.ENDC}        {Colors.YELLOW}{run_name}{Colors.ENDC}")
        logger.info(f"{Colors.BOLD}Max Epochs:{Colors.ENDC}       {Colors.CYAN}{self.max_epoch}{Colors.ENDC}")
        logger.info(f"{Colors.BOLD}Early Stop Patience:{Colors.ENDC} {Colors.CYAN}{self.patience}{Colors.ENDC}")
        logger.info(f"{Colors.BOLD}Output Directory:{Colors.ENDC} {Colors.YELLOW}{self.run_dir}{Colors.ENDC}")
        
        if self.hyperparams:
            logger.info(f"{Colors.BOLD}Hyperparameters:{Colors.ENDC}")
            for key, value in self.hyperparams.items():
                logger.info(f"  {Colors.CYAN}{key}:{Colors.ENDC} {value}")
        
        logger.info("")
    
    def train(self):
        """Train with tracking and detailed stopping reasons."""
        logger = logging.getLogger(__name__)
        
        try:
            super().train()
        except KeyboardInterrupt:
            self.stopping_reason = "User interrupted (Ctrl+C)"
            logger.warning(f"\nTraining interrupted: {self.stopping_reason}")
        except AssertionError as e:
            # Handle case where evaluation never ran during training
            if "No evaluation results" in str(e):
                self.stopping_reason = "Training completed but no evaluation occurred - check TEST.EVAL_PERIOD in config"
                logger.warning(f"\n{Colors.RED}{self.stopping_reason}{Colors.ENDC}")
                # Still save the results we have
            else:
                self.stopping_reason = f"AssertionError: {str(e)}"
                logger.error(f"\nTraining failed: {self.stopping_reason}")
                raise
        except Exception as e:
            self.stopping_reason = f"Error: {str(e)}"
            logger.error(f"\nTraining failed: {self.stopping_reason}")
            raise
        finally:
            self.save_stopping_reason()
            self.save_model_info()
            self.save_training_history()
            self.print_summary()
        
        if comm.is_main_process():
            # Return results if we have them, otherwise return empty dict
            if hasattr(self, "_last_eval_results"):
                return self._last_eval_results
            else:
                return {}
    
    def display_progress_bar(self, current_epoch, total_epochs):
        """Display a visual progress bar for training progress."""
        bar_length = 50
        progress = current_epoch / total_epochs
        filled = int(bar_length * progress)
        bar = '█' * filled + '░' * (bar_length - filled)
        percentage = int(progress * 100)
        
        progress_str = f"{Colors.BOLD}{Colors.CYAN}[{bar}] {percentage:3d}% ({current_epoch}/{total_epochs}){Colors.ENDC}"
        return progress_str
    
    def after_epoch(self):
        """Called after each epoch - properly capture all available metrics."""
        # CRITICAL: Call parent's after_epoch() to trigger hooks (including EvalHook)
        # This must happen FIRST before we check for results
        super().after_epoch()
        
        if not comm.is_main_process():
            return
        
        logger = logging.getLogger(__name__)
        epoch_num = self.epoch + 1
        
        # Display progress bar every epoch
        progress_bar = self.display_progress_bar(epoch_num, self.max_epoch)
        logger.info(f"")
        logger.info(f"{Colors.BOLD}{Colors.BLUE}Epoch {epoch_num}/{self.max_epoch}{Colors.ENDC} {progress_bar}")
        
        # Try multiple ways to get evaluation results
        current_mAP = 0.0
        current_top1 = 0.0
        has_eval = False
        
        # Method 1: Check _last_eval_results (set by evaluation hook)
        # Only use if it was just updated THIS epoch (not stale from previous evaluations)
        if hasattr(self, '_last_eval_results') and self._last_eval_results:
            # Check if this is fresh evaluation (should have mAP > 0 if just evaluated)
            current_mAP = float(self._last_eval_results.get('mAP', 0.0))
            current_top1 = float(self._last_eval_results.get('top-1', 0.0))
            
            # Only count as evaluation if EVAL_PERIOD says we should evaluate THIS epoch
            next_epoch = epoch_num
            if self.cfg.TEST.EVAL_PERIOD > 0 and next_epoch % self.cfg.TEST.EVAL_PERIOD == 0:
                # This epoch should have evaluation - results are fresh
                if current_mAP > 0.0:
                    has_eval = True
            # Otherwise results are stale from previous evaluation, don't use them
        
        # Method 2: Check storage (fallback)
        # Only use storage metrics if THIS epoch was scheduled for evaluation
        if not has_eval and hasattr(self, 'storage'):
            next_epoch = epoch_num
            # Only look at storage if this epoch should have evaluation
            if self.cfg.TEST.EVAL_PERIOD > 0 and next_epoch % self.cfg.TEST.EVAL_PERIOD == 0:
                storage = self.storage
                latest_metrics = storage.latest()
                if latest_metrics and 'mAP' in latest_metrics:
                    current_mAP = extract_metric_value(latest_metrics.get('mAP', 0.0))
                    current_top1 = extract_metric_value(latest_metrics.get('top-1', 0.0))
                    if current_mAP > 0.0:
                        has_eval = True
        
        # If we got evaluation results, log and track them
        if has_eval:
            map_color = Colors.GREEN if current_mAP > 0.3 else Colors.YELLOW if current_mAP > 0.1 else Colors.RED
            
            logger.info(f"  {Colors.BOLD}Metrics:{Colors.ENDC}")
            logger.info(f"    mAP:   {map_color}{current_mAP:.4f}{Colors.ENDC}")
            logger.info(f"    top-1: {Colors.CYAN}{current_top1:.4f}{Colors.ENDC}")
            
            # Track in history
            self.training_history['epoch'].append(epoch_num)
            self.training_history['mAP'].append(current_mAP)
            self.training_history['top1'].append(current_top1)
            
            # Generate metric plots after evaluation (only once per evaluation epoch)
            if PLOTTER_AVAILABLE:
                try:
                    history_file = Path(self.cfg.OUTPUT_DIR) / 'training_history.json'
                    if history_file.exists():
                        plot_output_dir = generate_all_plots(str(self.cfg.OUTPUT_DIR), str(history_file))
                        logger.info(f"  {Colors.BOLD}{Colors.CYAN}📊 Plots saved to: {plot_output_dir}{Colors.ENDC}")
                except Exception as e:
                    logger.debug(f"Plot generation skipped: {str(e)}")
            
            # Early stopping logic
            if current_mAP > self.best_mAP:
                self.best_mAP = current_mAP
                self.best_epoch = epoch_num
                self.epochs_without_improvement = 0
                logger.info(f"  {Colors.GREEN}{Colors.BOLD}✓ New best mAP!{Colors.ENDC}")
            else:
                self.epochs_without_improvement += 1
                logger.info(f"  {Colors.YELLOW}No improvement ({self.epochs_without_improvement}/{self.patience}){Colors.ENDC}")
                
                if self.epochs_without_improvement >= self.patience:
                    self.should_stop = True
                    self.stopping_reason = f"No mAP improvement for {self.patience} epochs"
                    logger.warning(f"  {Colors.RED}{Colors.BOLD}⏹ Stopping: {self.stopping_reason}{Colors.ENDC}")
        else:
            logger.info(f"  {Colors.CYAN}Training in progress (evaluating every {self.cfg.TEST.EVAL_PERIOD} epochs)...{Colors.ENDC}")
    
    def save_stopping_reason(self):
        """Save detailed stopping reason to file."""
        if not comm.is_main_process():
            return
        
        logger = logging.getLogger(__name__)
        
        content = f"""
╔════════════════════════════════════════════════════════════════════╗
║                    TRAINING STOPPING REASON                        ║
╚════════════════════════════════════════════════════════════════════╝

Run Name: {self.run_name}
Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

STOPPING REASON:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{self.stopping_reason}

TRAINING STATISTICS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Total Epochs Trained: {len(self.training_history['epoch'])}
  Maximum Epochs Allowed: {self.max_epoch}
  
  Best mAP: {self.best_mAP:.4f} (Epoch {self.best_epoch})
  Best top-1: {max(self.training_history['top1']) if self.training_history['top1'] else 0:.4f}
  
  Final mAP: {self.training_history['mAP'][-1] if self.training_history['mAP'] else 0:.4f}
  Final top-1: {self.training_history['top1'][-1] if self.training_history['top1'] else 0:.4f}

HYPERPARAMETERS USED:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
        for key, value in self.hyperparams.items():
            content += f"  {key}: {value}\n"
        
        content += f"""
RECOMMENDATION:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
        if self.best_mAP < 0.2:
            content += "  ⚠ Model performance is poor. Try:\n"
            content += "    - Increase learning rate (current may be too low)\n"
            content += "    - Train longer (increase MAX_EPOCH)\n"
            content += "    - Check dataset loading\n"
        elif self.best_mAP < 0.4:
            content += "  ℹ Model performance is below target. Try:\n"
            content += "    - Use different learning rate schedule\n"
            content += "    - Adjust loss weights (higher triplet loss weight)\n"
            content += "    - Use hard triplet mining\n"
        elif self.best_mAP < 0.5:
            content += "  ✓ Model is performing reasonably. Try:\n"
            content += "    - Fine-tune learning rate for further improvement\n"
            content += "    - Add regularization techniques\n"
        else:
            content += "  ✓✓ Model is performing well!\n"
            content += "    Consider deploying this model.\n"
        
        content += f"""
═════════════════════════════════════════════════════════════════════

Model files saved in: {self.run_dir}/
  - best_model.pth (best checkpoint)
  - model_final.pth (final checkpoint)
  - training_history.json (detailed metrics)
  - STOPPING_REASON.txt (this file)
  - model_info.json (metadata)
"""
        
        with open(self.stop_reason_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        logger.info(f"{Colors.BOLD}{Colors.GREEN}✓ Stopping reason saved to:{Colors.ENDC} {Colors.YELLOW}{self.stop_reason_file}{Colors.ENDC}")
    
    def save_model_info(self):
        """Save model metadata."""
        if not comm.is_main_process():
            return
        
        info = {
            'run_name': self.run_name,
            'timestamp': datetime.now().isoformat(),
            'stopping_reason': self.stopping_reason,
            'hyperparameters': self.hyperparams,
            'best_mAP': float(self.best_mAP),
            'best_epoch': int(self.best_epoch),
            'best_top1': float(max(self.training_history['top1'])) if self.training_history['top1'] else 0.0,
            'final_mAP': float(self.training_history['mAP'][-1]) if self.training_history['mAP'] else 0.0,
            'final_top1': float(self.training_history['top1'][-1]) if self.training_history['top1'] else 0.0,
            'epochs_trained': len(self.training_history['epoch']),
            'max_epochs': self.max_epoch,
        }
        
        with open(self.model_info_file, 'w', encoding='utf-8') as f:
            json.dump(info, f, indent=2)
    
    def save_training_history(self):
        """Save detailed training history."""
        if not comm.is_main_process():
            return
        
        with open(self.history_file, 'w', encoding='utf-8') as f:
            json.dump(self.training_history, f, indent=2)
    
    def print_summary(self):
        """Print final summary with colors."""
        if not comm.is_main_process():
            return
        
        logger = logging.getLogger(__name__)
        
        # Create colored summary
        logger.info("")
        logger.info(f"{Colors.BOLD}{Colors.CYAN}┏{'━'*68}┓{Colors.ENDC}")
        logger.info(f"{Colors.BOLD}{Colors.CYAN}┃{' '*20}TRAINING SUMMARY{' '*30}┃{Colors.ENDC}")
        logger.info(f"{Colors.BOLD}{Colors.CYAN}┗{'━'*68}┛{Colors.ENDC}")
        logger.info("")
        
        # Stopping reason with appropriate color
        reason_color = Colors.GREEN if "improvement" in self.stopping_reason.lower() else Colors.YELLOW
        logger.info(f"{Colors.BOLD}Stopping Reason:{Colors.ENDC}")
        logger.info(f"  {reason_color}{self.stopping_reason}{Colors.ENDC}")
        logger.info("")
        
        # Performance metrics
        map_color = Colors.GREEN if self.best_mAP > 0.3 else Colors.YELLOW
        logger.info(f"{Colors.BOLD}Performance:{Colors.ENDC}")
        logger.info(f"  Best mAP:    {map_color}{self.best_mAP:.4f}{Colors.ENDC} (Epoch {Colors.BLUE}{self.best_epoch}{Colors.ENDC})")
        if self.training_history['top1']:
            logger.info(f"  Best top-1:  {Colors.CYAN}{max(self.training_history['top1']):.4f}{Colors.ENDC}")
        if self.training_history['mAP']:
            logger.info(f"  Final mAP:   {Colors.YELLOW}{self.training_history['mAP'][-1]:.4f}{Colors.ENDC}")
        logger.info("")
        
        # Training info
        logger.info(f"{Colors.BOLD}Training Info:{Colors.ENDC}")
        logger.info(f"  Epochs Trained: {Colors.CYAN}{len(self.training_history['epoch'])}/{self.max_epoch}{Colors.ENDC}")
        logger.info(f"  Output Dir:     {Colors.YELLOW}{self.run_dir}{Colors.ENDC}")
        logger.info("")
        
        # Generated files
        logger.info(f"{Colors.BOLD}Output Files:{Colors.ENDC}")
        logger.info(f"  {Colors.GREEN}✓{Colors.ENDC} STOPPING_REASON.txt (detailed report)")
        logger.info(f"  {Colors.GREEN}✓{Colors.ENDC} model_info.json (metadata)")
        logger.info(f"  {Colors.GREEN}✓{Colors.ENDC} training_history.json (metrics history)")
        logger.info("")
        logger.info(f"{Colors.BOLD}{Colors.CYAN}┗{'━'*68}┛{Colors.ENDC}")
        logger.info("")


def setup(args):
    """Setup config and trainer."""
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    
    # Ensure early stopping config
    if not hasattr(cfg.SOLVER, 'EARLY_STOP_PATIENCE'):
        cfg.defrost()
        cfg.SOLVER.EARLY_STOP_PATIENCE = 10
        cfg.freeze()
    
    cfg.defrost()
    # Disable multiprocessing to avoid paging file issues on Windows during evaluation
    cfg.DATALOADER.NUM_WORKERS = 0
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)
    
    if args.eval_only:
        cfg.defrost()
        cfg.MODEL.BACKBONE.PRETRAIN = False
        model = ResearchTrainer.build_model(cfg)
        from fastreid.utils.checkpoint import Checkpointer
        Checkpointer(model).load(cfg.MODEL.WEIGHTS)
        res = ResearchTrainer.test(cfg, model)
        return res
    
    # Extract solution name from config file path
    # e.g., custom_configs/plateau_solutions/solution_1_higher_lr.yml -> solution_1_higher_lr
    config_path = args.config_file
    if 'plateau_solutions' in config_path:
        # Extract filename without .yml
        run_name = Path(config_path).stem
    else:
        run_name = args.run_name if hasattr(args, 'run_name') else 'default'
    
    trainer = ResearchTrainer(
        cfg,
        run_name=run_name,
        hyperparams={
            'max_epoch': cfg.SOLVER.MAX_EPOCH,
            'batch_size': cfg.SOLVER.IMS_PER_BATCH,
            'lr': cfg.SOLVER.BASE_LR,
            'patience': cfg.SOLVER.EARLY_STOP_PATIENCE,
        }
    )
    
    trainer.resume_or_load(resume=args.resume)
    results = trainer.train()
    
    # Generate metric plots after training completes
    if PLOTTER_AVAILABLE and comm.is_main_process():
        try:
            logger = logging.getLogger(__name__)
            run_dir = trainer.cfg.OUTPUT_DIR if hasattr(trainer, 'cfg') else None
            
            if run_dir:
                history_file = Path(run_dir) / 'training_history.json'
                if history_file.exists():
                    logger.info(f"\n{Colors.BOLD}{Colors.CYAN}Generating metric plots...{Colors.ENDC}")
                    plot_output_dir = generate_all_plots(str(run_dir), str(history_file))
                    logger.info(f"{Colors.BOLD}{Colors.GREEN}✓ Metric plots saved to: {plot_output_dir}{Colors.ENDC}\n")
                else:
                    logger.warning(f"Training history file not found: {history_file}")
            else:
                logger.warning("Cannot generate plots: OUTPUT_DIR not configured")
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.warning(f"Failed to generate metric plots: {str(e)}")
    
    return results


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    
    # Add run name argument
    import argparse
    if not hasattr(args, 'run_name'):
        args.run_name = 'default'
    
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
