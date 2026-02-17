#!/usr/bin/env python
# encoding: utf-8
"""
Model Comparison & Selection Tool for Researchers

This tool analyzes multiple training runs and identifies the best model
based on various criteria and metrics.
"""

import os
import json
import argparse
from pathlib import Path
from collections import defaultdict
import statistics


class ModelComparator:
    """Compare multiple training runs and select best model."""
    
    def __init__(self):
        self.runs = {}
    
    def add_run(self, run_name, output_dir):
        """Add a training run to comparison."""
        
        # Check if directory exists
        if not os.path.exists(output_dir):
            print(f"ERROR: Directory not found: {output_dir}")
            return False
        
        # Load validation history
        history_file = os.path.join(output_dir, 'validation_history.json')
        stopping_report = os.path.join(output_dir, 'STOPPING_REPORT.txt')
        
        if not os.path.exists(history_file):
            print(f"ERROR: No validation_history.json in {output_dir}")
            return False
        
        with open(history_file, 'r') as f:
            history = json.load(f)
        
        stopping_reason = None
        if os.path.exists(stopping_report):
            with open(stopping_report, 'r') as f:
                stopping_reason = f.read()
        
        self.runs[run_name] = {
            'output_dir': output_dir,
            'history': history,
            'stopping_reason': stopping_reason,
            'best_model_path': os.path.join(output_dir, 'best_model.pth'),
        }
        
        return True
    
    def compute_metrics(self, run_name):
        """Compute detailed metrics for a run."""
        run = self.runs[run_name]
        history = run['history']
        
        mAPs = history['mAP']
        top1s = history['top1']
        epochs = history['epoch']
        losses = history['train_loss']
        overfitting_flags = history.get('overfitting_flag', [False] * len(epochs))
        
        metrics = {
            'name': run_name,
            'num_epochs': len(epochs),
            'best_epoch': epochs[mAPs.index(max(mAPs))],
            'best_mAP': max(mAPs),
            'final_mAP': mAPs[-1],
            'mAP_improvement': max(mAPs) - mAPs[0],
            'mAP_improvement_pct': (max(mAPs) - mAPs[0]) / mAPs[0] * 100 if mAPs[0] > 0 else 0,
            'best_top1': max(top1s),
            'final_top1': top1s[-1],
            'initial_loss': losses[0],
            'final_loss': losses[-1],
            'loss_reduction': losses[0] - losses[-1],
            'mAP_stability': statistics.stdev(mAPs) if len(mAPs) > 1 else 0,
            'overfitting_count': sum(overfitting_flags),
            'overfitting_pct': sum(overfitting_flags) / len(overfitting_flags) * 100 if overfitting_flags else 0,
        }
        
        # Check if model is still improving at end
        last_5_mAPs = mAPs[-5:] if len(mAPs) >= 5 else mAPs
        metrics['still_improving'] = all(last_5_mAPs[i] <= last_5_mAPs[i+1] for i in range(len(last_5_mAPs)-1))
        
        return metrics
    
    def print_comparison_table(self):
        """Print formatted comparison table."""
        if not self.runs:
            print("No runs to compare!")
            return
        
        all_metrics = {name: self.compute_metrics(name) for name in self.runs}
        
        print("\n" + "="*140)
        print("MODEL COMPARISON TABLE")
        print("="*140)
        print(f"{'Model':<20} {'Epochs':<10} {'Best mAP':<12} {'Final mAP':<12} {'Improvement':<15} {'Top-1':<10} {'Loss Red.':<12}")
        print("-"*140)
        
        for name, metrics in sorted(all_metrics.items()):
            improvement = f"{metrics['mAP_improvement_pct']:+.1f}%"
            loss_red = f"{metrics['loss_reduction']:.2f}"
            
            print(f"{name:<20} {metrics['num_epochs']:<10} "
                  f"{metrics['best_mAP']:<12.4f} {metrics['final_mAP']:<12.4f} "
                  f"{improvement:<15} {metrics['best_top1']:<10.4f} {loss_red:<12}")
        
        print("="*140 + "\n")
    
    def print_detailed_analysis(self, run_name):
        """Print detailed analysis of a specific run."""
        if run_name not in self.runs:
            print(f"ERROR: Run '{run_name}' not found")
            return
        
        metrics = self.compute_metrics(run_name)
        run = self.runs[run_name]
        
        print("\n" + "="*80)
        print(f"DETAILED ANALYSIS: {run_name}")
        print("="*80)
        
        # Overview
        print("\n📊 OVERVIEW")
        print("-"*80)
        print(f"Total Epochs Trained: {metrics['num_epochs']}")
        print(f"Best Epoch: {metrics['best_epoch']}")
        print(f"Output Directory: {run['output_dir']}")
        print(f"Best Model Path: {run['best_model_path']}")
        
        # mAP Metrics
        print("\n📈 mAP METRICS")
        print("-"*80)
        print(f"Initial mAP:      {run['history']['mAP'][0]:.4f}")
        print(f"Best mAP:         {metrics['best_mAP']:.4f} (epoch {metrics['best_epoch']})")
        print(f"Final mAP:        {metrics['final_mAP']:.4f}")
        print(f"Improvement:      {metrics['mAP_improvement']:+.4f} ({metrics['mAP_improvement_pct']:+.1f}%)")
        print(f"Stability:        {metrics['mAP_stability']:.6f} (lower = more stable)")
        
        # Top-1 Metrics
        print("\n🎯 TOP-1 ACCURACY METRICS")
        print("-"*80)
        print(f"Initial Top-1:    {run['history']['top1'][0]:.4f}")
        print(f"Best Top-1:       {metrics['best_top1']:.4f}")
        print(f"Final Top-1:      {metrics['final_top1']:.4f}")
        
        # Loss Analysis
        print("\n📉 TRAINING LOSS ANALYSIS")
        print("-"*80)
        print(f"Initial Loss:     {metrics['initial_loss']:.4f}")
        print(f"Final Loss:       {metrics['final_loss']:.4f}")
        print(f"Reduction:        {metrics['loss_reduction']:+.4f}")
        
        # Overfitting Analysis
        print("\n⚠️  OVERFITTING ANALYSIS")
        print("-"*80)
        print(f"Overfitting Epochs: {metrics['overfitting_count']}")
        print(f"Percentage:       {metrics['overfitting_pct']:.1f}%")
        if metrics['overfitting_count'] > 0:
            print(f"Status:           ⚠ OVERFITTING DETECTED")
        else:
            print(f"Status:           ✓ NO OVERFITTING")
        
        # Training Status
        print("\n🎓 TRAINING STATUS")
        print("-"*80)
        if metrics['still_improving']:
            print(f"Status:           📈 STILL IMPROVING")
            print(f"Recommendation:   Consider training longer (increase MAX_EPOCH)")
        else:
            print(f"Status:           ✓ PLATEAU REACHED")
            print(f"Recommendation:   Training complete, use best_model.pth")
        
        print("\n" + "="*80)
    
    def select_best_model(self, criteria='best_mAP'):
        """Select best model based on criteria."""
        if not self.runs:
            print("No runs to compare!")
            return None
        
        all_metrics = {name: self.compute_metrics(name) for name in self.runs}
        
        print(f"\n🏆 SELECTING BEST MODEL (criteria: {criteria})")
        print("="*80)
        
        if criteria == 'best_mAP':
            best_run = max(all_metrics.items(), key=lambda x: x[1]['best_mAP'])
        elif criteria == 'final_mAP':
            best_run = max(all_metrics.items(), key=lambda x: x[1]['final_mAP'])
        elif criteria == 'improvement':
            best_run = max(all_metrics.items(), key=lambda x: x[1]['mAP_improvement'])
        elif criteria == 'stability':
            best_run = min(all_metrics.items(), key=lambda x: x[1]['mAP_stability'])
        elif criteria == 'no_overfitting':
            # Prefer models with no overfitting, then highest mAP
            best_run = min(all_metrics.items(), 
                          key=lambda x: (x[1]['overfitting_count'], -x[1]['best_mAP']))
        else:
            print(f"Unknown criteria: {criteria}")
            return None
        
        name, metrics = best_run
        
        print(f"\n✅ WINNER: {name}")
        print("-"*80)
        print(f"Best mAP:        {metrics['best_mAP']:.4f}")
        print(f"Improvement:     {metrics['mAP_improvement_pct']:+.1f}%")
        print(f"Overfitting:     {metrics['overfitting_count']} epochs flagged")
        print(f"Path:            {self.runs[name]['best_model_path']}")
        print("\n" + "="*80)
        
        return name, metrics
    
    def generate_report(self, output_file=None):
        """Generate comprehensive comparison report."""
        if not self.runs:
            print("No runs to compare!")
            return
        
        if output_file is None:
            output_file = "model_comparison_report.txt"
        
        with open(output_file, 'w') as f:
            f.write("="*100 + "\n")
            f.write("MODEL TRAINING COMPARISON REPORT\n")
            f.write("="*100 + "\n\n")
            
            # Summary table
            f.write("SUMMARY TABLE\n")
            f.write("-"*100 + "\n")
            
            all_metrics = {name: self.compute_metrics(name) for name in self.runs}
            
            f.write(f"{'Model':<25} {'Epochs':<10} {'Best mAP':<12} {'Final mAP':<12} "
                   f"{'Improvement':<15} {'Overfitting':<15}\n")
            f.write("-"*100 + "\n")
            
            for name, metrics in sorted(all_metrics.items()):
                overfitting = f"{metrics['overfitting_count']} epochs" if metrics['overfitting_count'] > 0 else "None"
                improvement = f"{metrics['mAP_improvement_pct']:+.1f}%"
                
                f.write(f"{name:<25} {metrics['num_epochs']:<10} "
                       f"{metrics['best_mAP']:<12.4f} {metrics['final_mAP']:<12.4f} "
                       f"{improvement:<15} {overfitting:<15}\n")
            
            # Detailed analysis for each run
            f.write("\n" + "="*100 + "\n")
            f.write("DETAILED ANALYSIS PER MODEL\n")
            f.write("="*100 + "\n\n")
            
            for run_name in sorted(self.runs.keys()):
                metrics = all_metrics[run_name]
                run = self.runs[run_name]
                history = run['history']
                
                f.write(f"\n{'='*100}\n")
                f.write(f"MODEL: {run_name}\n")
                f.write(f"{'='*100}\n")
                
                f.write(f"Output Directory: {run['output_dir']}\n")
                f.write(f"Best Model: {run['best_model_path']}\n\n")
                
                f.write(f"mAP: {history['mAP'][0]:.4f} → {metrics['best_mAP']:.4f} → {history['mAP'][-1]:.4f}\n")
                f.write(f"Top-1: {history['top1'][0]:.4f} → {metrics['best_top1']:.4f} → {history['top1'][-1]:.4f}\n")
                f.write(f"Loss: {metrics['initial_loss']:.4f} → {metrics['final_loss']:.4f}\n")
                f.write(f"Overfitting Flags: {metrics['overfitting_count']}/{metrics['num_epochs']}\n")
                f.write(f"Improvement: {metrics['mAP_improvement_pct']:+.1f}%\n")
                
                if metrics['still_improving']:
                    f.write(f"Status: STILL IMPROVING (consider longer training)\n")
                else:
                    f.write(f"Status: PLATEAU REACHED\n")
        
        print(f"\nReport saved to: {output_file}")
        print(f"Open this file for detailed comparison: {os.path.abspath(output_file)}")


def main():
    parser = argparse.ArgumentParser(
        description='Compare multiple ReID training runs and select best model'
    )
    parser.add_argument('run_dirs', nargs='+', 
                       help='Directory paths from training runs (each should have validation_history.json)')
    parser.add_argument('--names', nargs='+',
                       help='Custom names for runs (must match number of run_dirs)')
    parser.add_argument('--criteria', choices=['best_mAP', 'final_mAP', 'improvement', 'stability', 'no_overfitting'],
                       default='best_mAP',
                       help='Criteria for selecting best model')
    parser.add_argument('--report', '-r', action='store_true',
                       help='Generate comprehensive report file')
    parser.add_argument('--output', '-o',
                       help='Output file for report')
    
    args = parser.parse_args()
    
    # Create comparator
    comparator = ModelComparator()
    
    # Add runs
    names = args.names if args.names else [f"Run_{i+1}" for i in range(len(args.run_dirs))]
    
    if len(names) != len(args.run_dirs):
        print(f"ERROR: Number of names ({len(names)}) doesn't match number of directories ({len(args.run_dirs)})")
        return
    
    print("Loading training runs...")
    for name, run_dir in zip(names, args.run_dirs):
        if comparator.add_run(name, run_dir):
            print(f"  ✓ {name}: {run_dir}")
        else:
            print(f"  ✗ {name}: {run_dir}")
    
    # Print comparison
    comparator.print_comparison_table()
    
    # Print detailed analysis for first run
    if names:
        comparator.print_detailed_analysis(names[0])
    
    # Select best
    best_run = comparator.select_best_model(args.criteria)
    
    # Generate report if requested
    if args.report:
        comparator.generate_report(args.output)


if __name__ == "__main__":
    main()
