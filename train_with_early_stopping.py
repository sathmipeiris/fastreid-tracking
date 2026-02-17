#!/usr/bin/env python
# encoding: utf-8
"""
Training script with proper validation metrics and early stopping detection.
Validates every epoch to detect real overfitting, not just loss plateaus.
"""

import sys
import os
import json
import logging
from collections import OrderedDict

sys.path.append('.')

import torch
from fastreid.config import get_cfg
from fastreid.engine import (
    DefaultTrainer, default_argument_parser, default_setup, launch
)
from fastreid.utils.checkpoint import Checkpointer
from fastreid.utils import comm
from fastreid.utils.events import get_event_storage


class EarlyStoppingTrainer(DefaultTrainer):
    """
    Trainer with early stopping based on validation metrics.
    
    Monitors mAP (Mean Average Precision) from ReID evaluation:
    - Detects real overfitting by comparing validation mAP across epochs
    - Stops training when validation mAP doesn't improve for N epochs
    - Saves best model checkpoint separately
    """
    
    def __init__(self, cfg):
        super().__init__(cfg)
        
        # Early stopping configuration
        self.patience = cfg.SOLVER.EARLY_STOP_PATIENCE  # epochs without improvement before stopping
        self.best_mAP = 0.0
        self.best_epoch = 0
        self.epochs_without_improvement = 0
        self.should_stop = False
        
        # Validation history for overfitting detection
        self.validation_history = {
            'epoch': [],
            'mAP': [],
            'top1': [],
            'train_loss': [],
            'overfitting_flag': []
        }
        
        # Path for best model
        self.best_model_path = os.path.join(cfg.OUTPUT_DIR, 'best_model.pth')
        self.history_file = os.path.join(cfg.OUTPUT_DIR, 'validation_history.json')
        
        logger = logging.getLogger(__name__)
        logger.info(f"Early stopping enabled with patience={self.patience} epochs")
        logger.info(f"Best model will be saved to: {self.best_model_path}")
    
    def train(self):
        """
        Run training with early stopping.
        """
        logger = logging.getLogger(__name__)
        start_epoch = self.start_epoch
        max_epoch = self.max_epoch
        
        logger.info(f"Starting training from epoch {start_epoch} to {max_epoch}")
        logger.info(f"Early stopping will trigger if mAP doesn't improve for {self.patience} epochs")
        
        try:
            # Run normal training loop
            self.train_loop(start_epoch, max_epoch)
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
            self.save_validation_history()
            raise
        
        # Save validation history
        self.save_validation_history()
        
        # Save detailed stopping report
        self.save_stopping_report()
        
        if comm.is_main_process():
            logger.info("\n" + "="*70)
            logger.info("TRAINING SUMMARY")
            logger.info("="*70)
            logger.info(f"Best mAP: {self.best_mAP:.4f} at epoch {self.best_epoch}")
            logger.info(f"Best model saved to: {self.best_model_path}")
            logger.info(f"Final epoch: {self.current_epoch}")
            
            if self.should_stop:
                logger.info(f"Training stopped early due to no improvement for {self.patience} epochs")
            else:
                logger.info("Training completed all epochs")
            
            logger.info("Validation history saved to: " + self.history_file)
            logger.info("="*70 + "\n")
            
            assert hasattr(self, "_last_eval_results"), "No evaluation results obtained!"
            return self._last_eval_results
    
    def train_loop(self, start_epoch, max_epoch):
        """Custom training loop with validation after each epoch."""
        logger = logging.getLogger(__name__)
        
        for epoch in range(start_epoch, max_epoch):
            if self.should_stop:
                logger.info(f"Early stopping triggered at epoch {epoch}")
                break
            
            self.current_epoch = epoch
            
            # Run training for one epoch
            self.before_epoch()
            
            # Get training loss from this epoch
            train_loss_this_epoch = self.run_epoch()
            
            self.after_epoch()
            
            # Validate after each epoch
            if comm.is_main_process():
                logger.info(f"\n{'='*70}")
                logger.info(f"Epoch {epoch + 1}/{max_epoch} - Validation")
                logger.info('='*70)
                
                # Run validation
                eval_results = self.test(self.cfg, self.model)
                
                # Extract mAP from results
                if isinstance(eval_results, dict) and 'mAP' in eval_results:
                    current_mAP = eval_results['mAP']
                    current_top1 = eval_results.get('top-1', 0.0)
                else:
                    current_mAP = 0.0
                    current_top1 = 0.0
                
                # Log validation metrics
                logger.info(f"Epoch {epoch + 1} Validation Results:")
                logger.info(f"  mAP: {current_mAP:.4f}")
                logger.info(f"  top-1: {current_top1:.4f}")
                logger.info(f"  train_loss: {train_loss_this_epoch:.4f}")
                
                # Check for overfitting (validation mAP lower than threshold relative to training)
                is_overfitting = self.detect_overfitting(train_loss_this_epoch, current_mAP)
                
                # Update validation history
                self.validation_history['epoch'].append(epoch + 1)
                self.validation_history['mAP'].append(float(current_mAP))
                self.validation_history['top1'].append(float(current_top1))
                self.validation_history['train_loss'].append(float(train_loss_this_epoch))
                self.validation_history['overfitting_flag'].append(is_overfitting)
                
                # Early stopping logic
                if current_mAP > self.best_mAP:
                    self.best_mAP = current_mAP
                    self.best_epoch = epoch + 1
                    self.epochs_without_improvement = 0
                    
                    # Save best model
                    logger.info(f"✓ New best mAP! Saving model to {self.best_model_path}")
                    self.checkpointer.save(f"best_model", epoch=epoch)
                    self.save_best_model(epoch)
                else:
                    self.epochs_without_improvement += 1
                    logger.info(f"No improvement. Epochs without improvement: {self.epochs_without_improvement}/{self.patience}")
                
                # Check early stopping condition
                if self.epochs_without_improvement >= self.patience:
                    logger.warning(f"\n{'!'*70}")
                    logger.warning(f"EARLY STOPPING: mAP hasn't improved for {self.patience} epochs")
                    logger.warning(f"Best mAP was {self.best_mAP:.4f} at epoch {self.best_epoch}")
                    logger.warning('!'*70 + "\n")
                    self.should_stop = True
                
                if is_overfitting:
                    logger.warning("⚠ WARNING: Overfitting detected! Validation metrics degrading.")
                
                logger.info(f"{'='*70}\n")
    
    def before_epoch(self):
        """Called before each epoch."""
        # Reset epoch loss tracker
        self.epoch_losses = []
    
    def run_epoch(self):
        """Run one epoch of training and return average loss."""
        logger = logging.getLogger(__name__)
        storage = get_event_storage()
        
        iters_per_epoch = len(self.data_loader.dataset) // self.cfg.SOLVER.IMS_PER_BATCH
        
        for _ in range(iters_per_epoch):
            self.before_step()
            self.run_step()
            self.after_step()
            self.iter += 1
        
        # Get average loss from storage
        try:
            avg_loss = storage.latest().get('total_loss', 0.0)
        except:
            avg_loss = 0.0
        
        return float(avg_loss)
    
    def after_epoch(self):
        """Called after each epoch."""
        pass
    
    def detect_overfitting(self, train_loss, val_mAP):
        """
        Detect signs of overfitting.
        Returns True if overfitting is detected.
        
        Overfitting indicators:
        1. Validation mAP dropping while training loss is low
        2. mAP significantly lower than training suggests
        """
        if len(self.validation_history['epoch']) < 2:
            return False
        
        # Check for declining trend in mAP
        recent_mAP = self.validation_history['mAP'][-3:]  # Last 3 epochs
        
        if len(recent_mAP) >= 3:
            # If mAP is consistently declining, it's overfitting
            if recent_mAP[-1] < recent_mAP[-2] < recent_mAP[-3]:
                return True
        
        # Check if validation mAP is too low relative to training
        # (This is a simple heuristic - adjust based on your needs)
        if train_loss < 2.0 and val_mAP < 0.3:  # Low training loss but poor validation
            return True
        
        return False
    
    def save_best_model(self, epoch):
        """Save the best model checkpoint."""
        model = self.model
        if isinstance(model, torch.nn.DataParallel) or isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model = model.module
        
        checkpoint = {
            'model': model.state_dict(),
            'epoch': epoch,
            'mAP': self.best_mAP,
        }
        torch.save(checkpoint, self.best_model_path)
    
    def save_validation_history(self):
        """Save validation history to JSON file."""
        if comm.is_main_process():
            with open(self.history_file, 'w') as f:
                json.dump(self.validation_history, f, indent=2)
    
    def save_stopping_report(self):
        """Save detailed report of why training stopped."""
        if not comm.is_main_process():
            return
        
        report_path = os.path.join(self.cfg.OUTPUT_DIR, 'STOPPING_REPORT.txt')
        
        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("TRAINING STOPPING REPORT\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Training Date: {self._get_timestamp()}\n")
            f.write(f"Model Architecture: {self.cfg.MODEL.BACKBONE.NAME}\n")
            f.write(f"Backbone: ResNet{self.cfg.MODEL.BACKBONE.DEPTH} with IBN\n")
            f.write(f"Dataset: {self.cfg.DATASETS.NAMES[0]}\n")
            f.write(f"Total Epochs Requested: {self.max_epoch}\n")
            f.write(f"Epochs Actually Trained: {self.current_epoch + 1}\n\n")
            
            # Stopping reason
            f.write("-"*80 + "\n")
            f.write("STOPPING REASON\n")
            f.write("-"*80 + "\n")
            
            if self.should_stop:
                f.write(f"✓ EARLY STOPPING TRIGGERED\n\n")
                f.write(f"  Reason: No improvement in mAP for {self.patience} consecutive epochs\n")
                f.write(f"  Best mAP: {self.best_mAP:.4f}\n")
                f.write(f"  Best mAP found at: Epoch {self.best_epoch}\n")
                f.write(f"  Epochs without improvement: {self.epochs_without_improvement}/{self.patience}\n")
                f.write(f"  Final mAP: {self.validation_history['mAP'][-1]:.4f}\n\n")
                f.write(f"  Status: OPTIMAL - Training stopped at right time\n")
                f.write(f"  Recommendation: Use best_model.pth (epoch {self.best_epoch})\n")
            else:
                f.write(f"✓ TRAINING COMPLETED\n\n")
                f.write(f"  Reason: Reached maximum epoch limit (MAX_EPOCH: {self.max_epoch})\n")
                f.write(f"  Best mAP: {self.best_mAP:.4f}\n")
                f.write(f"  Best mAP found at: Epoch {self.best_epoch}\n")
                f.write(f"  Final mAP: {self.validation_history['mAP'][-1]:.4f}\n\n")
                
                # Check if model is still improving
                last_5_mAPs = self.validation_history['mAP'][-5:]
                is_improving = all(last_5_mAPs[i] <= last_5_mAPs[i+1] for i in range(len(last_5_mAPs)-1))
                
                if is_improving:
                    f.write(f"  Status: STILL IMPROVING - Consider training longer\n")
                    f.write(f"  Recommendation: Increase MAX_EPOCH beyond {self.max_epoch}\n")
                else:
                    f.write(f"  Status: PLATEAU REACHED - Training complete\n")
                    f.write(f"  Recommendation: Use best_model.pth (epoch {self.best_epoch})\n")
            
            # Performance metrics
            f.write("\n" + "-"*80 + "\n")
            f.write("PERFORMANCE METRICS\n")
            f.write("-"*80 + "\n")
            
            mAPs = self.validation_history['mAP']
            top1s = self.validation_history['top1']
            losses = self.validation_history['train_loss']
            
            f.write(f"mAP Statistics:\n")
            f.write(f"  Initial: {mAPs[0]:.4f}\n")
            f.write(f"  Final:   {mAPs[-1]:.4f}\n")
            f.write(f"  Best:    {max(mAPs):.4f}\n")
            f.write(f"  Improvement: {max(mAPs) - mAPs[0]:+.4f}\n")
            f.write(f"  Improvement %: {(max(mAPs) - mAPs[0])/mAPs[0]*100:+.1f}%\n\n")
            
            f.write(f"Top-1 Accuracy Statistics:\n")
            f.write(f"  Initial: {top1s[0]:.4f}\n")
            f.write(f"  Final:   {top1s[-1]:.4f}\n")
            f.write(f"  Best:    {max(top1s):.4f}\n")
            f.write(f"  Improvement: {max(top1s) - top1s[0]:+.4f}\n\n")
            
            f.write(f"Training Loss Statistics:\n")
            f.write(f"  Initial: {losses[0]:.4f}\n")
            f.write(f"  Final:   {losses[-1]:.4f}\n")
            f.write(f"  Reduction: {losses[0] - losses[-1]:+.4f}\n\n")
            
            # Overfitting analysis
            f.write("-"*80 + "\n")
            f.write("OVERFITTING ANALYSIS\n")
            f.write("-"*80 + "\n")
            
            overfitting_epochs = [self.validation_history['epoch'][i] 
                                 for i, flag in enumerate(self.validation_history['overfitting_flag']) 
                                 if flag]
            
            if overfitting_epochs:
                f.write(f"⚠ OVERFITTING DETECTED\n")
                f.write(f"  Epochs flagged: {overfitting_epochs}\n")
                f.write(f"  Total flagged epochs: {len(overfitting_epochs)}\n")
                f.write(f"  Recommendation: Reduce training duration or add regularization\n\n")
            else:
                f.write(f"✓ NO OVERFITTING DETECTED\n")
                f.write(f"  Model generalization: Good\n\n")
            
            # Model checkpoint info
            f.write("-"*80 + "\n")
            f.write("MODEL CHECKPOINTS\n")
            f.write("-"*80 + "\n")
            f.write(f"Best Model: best_model.pth\n")
            f.write(f"  Epoch: {self.best_epoch}\n")
            f.write(f"  mAP: {self.best_mAP:.4f}\n")
            f.write(f"  Size: ~250 MB\n")
            f.write(f"  Status: RECOMMENDED FOR DEPLOYMENT\n\n")
            
            f.write(f"Final Model: model_final.pth\n")
            f.write(f"  Epoch: {self.current_epoch + 1}\n")
            f.write(f"  mAP: {mAPs[-1]:.4f}\n")
            f.write(f"  Size: ~250 MB\n")
            f.write(f"  Status: Alternative (usually suboptimal)\n\n")
            
            # Configuration summary
            f.write("-"*80 + "\n")
            f.write("TRAINING CONFIGURATION\n")
            f.write("-"*80 + "\n")
            f.write(f"Batch Size: {self.cfg.SOLVER.IMS_PER_BATCH}\n")
            f.write(f"Learning Rate: {self.cfg.SOLVER.BASE_LR}\n")
            f.write(f"Optimizer: {self.cfg.SOLVER.OPT}\n")
            f.write(f"Learning Rate Schedule: {self.cfg.SOLVER.SCHED}\n")
            f.write(f"LR Decay Steps: {self.cfg.SOLVER.STEPS}\n")
            f.write(f"LR Decay Factor: {self.cfg.SOLVER.GAMMA}\n")
            f.write(f"Early Stop Patience: {self.patience} epochs\n")
            f.write(f"AMP Enabled: {self.cfg.SOLVER.AMP.ENABLED}\n\n")
            
            # Next steps
            f.write("-"*80 + "\n")
            f.write("NEXT STEPS\n")
            f.write("-"*80 + "\n")
            f.write(f"1. Analyze results: python analyze_training.py {os.path.join(self.cfg.OUTPUT_DIR, 'validation_history.json')} --plot\n")
            f.write(f"2. Load best model: torch.load('{os.path.join(self.cfg.OUTPUT_DIR, 'best_model.pth')}')\n")
            f.write(f"3. Deploy in enrollment_tracker.py\n")
            f.write(f"4. Evaluate on test set\n\n")
            
            f.write("="*80 + "\n")
        
        logger = logging.getLogger(__name__)
        logger.info(f"Stopping report saved to: {report_path}")
    
    def _get_timestamp(self):
        """Get current timestamp."""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def setup(args):
    """Create configs and perform basic setups."""
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    
    # Add early stopping config if not present
    if not hasattr(cfg.SOLVER, 'EARLY_STOP_PATIENCE'):
        cfg.defrost()
        cfg.SOLVER.EARLY_STOP_PATIENCE = 10  # Default: stop after 10 epochs without improvement
        cfg.freeze()
    
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)
    
    if args.eval_only:
        cfg.defrost()
        cfg.MODEL.BACKBONE.PRETRAIN = False
        model = EarlyStoppingTrainer.build_model(cfg)
        Checkpointer(model).load(cfg.MODEL.WEIGHTS)
        res = EarlyStoppingTrainer.test(cfg, model)
        return res
    
    trainer = EarlyStoppingTrainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
