#!/usr/bin/env python
# encoding: utf-8
"""
Researcher's Model Selection Tool
Automatically tests multiple hyperparameter configurations and selects the best model.
"""

import os
import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime
import argparse


class ResearchExperiment:
    """Manage multiple training runs with different hyperparameters."""
    
    def __init__(self, base_output_dir):
        self.base_dir = base_output_dir
        self.runs = []
        self.results_file = os.path.join(base_output_dir, 'EXPERIMENT_RESULTS.json')
        self.best_model_file = os.path.join(base_output_dir, 'BEST_MODEL_SELECTION.txt')
    
    def add_run(self, run_name, config_file, hyperparams):
        """Add a training run configuration."""
        self.runs.append({
            'name': run_name,
            'config': config_file,
            'params': hyperparams,
            'status': 'pending',
        })
    
    def run_training(self, run_name, config_file, opts=None):
        """Execute a single training run."""
        print(f"\n{'='*70}")
        print(f"STARTING RUN: {run_name}")
        print(f"{'='*70}\n")
        
        cmd = [
            sys.executable,
            'train_research_grade.py',
            '--config-file', config_file,
            '--run-name', run_name,
        ]
        
        if opts:
            cmd.extend(opts)
        
        try:
            result = subprocess.run(cmd, cwd=os.getcwd(), check=True)
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Run failed: {run_name}")
            return False
    
    def load_run_results(self, run_name, output_dir):
        """Load results from a completed training run."""
        run_dir = os.path.join(output_dir, run_name)
        info_file = os.path.join(run_dir, 'model_info.json')
        
        if not os.path.exists(info_file):
            return None
        
        with open(info_file, 'r') as f:
            return json.load(f)
    
    def compare_runs(self, output_dir):
        """Compare all completed runs and select best model."""
        print(f"\n{'='*70}")
        print("COMPARING ALL RUNS")
        print(f"{'='*70}\n")
        
        all_results = []
        
        for run in self.runs:
            results = self.load_run_results(run['name'], output_dir)
            if results:
                all_results.append(results)
                print(f"✓ {run['name']}")
                print(f"  mAP: {results['best_mAP']:.4f}")
                print(f"  Stopping: {results['stopping_reason']}")
                print()
        
        if not all_results:
            print("❌ No completed runs found!")
            return None
        
        # Find best by mAP
        best = max(all_results, key=lambda x: x['best_mAP'])
        
        # Find best by improvement
        best_improvement = max(all_results, 
                               key=lambda x: x['best_mAP'] - x.get('final_mAP', x['best_mAP']))
        
        return {
            'best_by_mAP': best,
            'best_by_improvement': best_improvement,
            'all_results': all_results,
        }
    
    def save_best_model_report(self, comparison_results, output_dir):
        """Generate comprehensive report on best model."""
        if not comparison_results:
            return
        
        best = comparison_results['best_by_mAP']
        
        report = f"""
╔════════════════════════════════════════════════════════════════════╗
║                  BEST MODEL SELECTION REPORT                       ║
╚════════════════════════════════════════════════════════════════════╝

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

WINNER: {best['run_name']}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Performance Metrics:
  • Best mAP: {best['best_mAP']:.4f}
  • Best top-1: {best['best_top1']:.4f}
  • Best epoch: {best['best_epoch']}
  • Epochs trained: {best['epochs_trained']}/{best['max_epochs']}

Hyperparameters Used:
"""
        for key, value in best['hyperparameters'].items():
            report += f"  • {key}: {value}\n"
        
        report += f"""
Stopping Reason:
  {best['stopping_reason']}

Model Location:
  Directory: {os.path.join(output_dir, best['run_name'])}
  Best Model: best_model.pth
  Final Model: model_final.pth
  Metadata: model_info.json
  Stopping Details: STOPPING_REASON.txt
  History: training_history.json

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

COMPARISON WITH OTHER RUNS:
"""
        
        for result in sorted(comparison_results['all_results'], 
                            key=lambda x: x['best_mAP'], 
                            reverse=True):
            delta = result['best_mAP'] - best['best_mAP']
            status = "⭐ WINNER" if result == best else "✓" if delta > -0.01 else "✗"
            report += f"\n{status} {result['run_name']}\n"
            report += f"   mAP: {result['best_mAP']:.4f} ({delta:+.4f})\n"
            report += f"   Stopping: {result['stopping_reason']}\n"
        
        report += f"""

═════════════════════════════════════════════════════════════════════

NEXT STEPS:
1. Review the best model's STOPPING_REASON.txt
2. Load best_model.pth from the winning run
3. Deploy using: enrollment_tracker.py --model <path>
4. Monitor performance on real data

To replicate this result:
  python train_research_grade.py \\
    --config-file <winning-config> \\
    --run-name "{best['run_name']}_replica"
"""
        
        with open(self.best_model_file, 'w') as f:
            f.write(report)
        
        print("\n" + "="*70)
        print(report)
        print("="*70)
        print(f"\nReport saved to: {self.best_model_file}")


def create_search_configs(base_config_path, output_dir):
    """Create multiple config files with different hyperparameters."""
    
    # Read base config
    with open(base_config_path, 'r') as f:
        base_content = f.read()
    
    # Define hyperparameter variations
    variations = [
        {
            'name': 'baseline',
            'description': 'Original configuration',
            'changes': {}
        },
        {
            'name': 'higher_lr',
            'description': 'Increased learning rate (0.0005 vs 0.00035)',
            'changes': {
                'BASE_LR: 0.00035': 'BASE_LR: 0.0005'
            }
        },
        {
            'name': 'lower_lr',
            'description': 'Decreased learning rate (0.0002 vs 0.00035)',
            'changes': {
                'BASE_LR: 0.00035': 'BASE_LR: 0.0002'
            }
        },
        {
            'name': 'higher_triplet_weight',
            'description': 'Increased triplet loss weight',
            'changes': {
                'SCALE: 1.': 'SCALE: 1.5  # Increased triplet weight'
            }
        },
        {
            'name': 'longer_training',
            'description': 'Extended to 120 epochs',
            'changes': {
                'MAX_EPOCH: 60': 'MAX_EPOCH: 120'
            }
        },
        {
            'name': 'smaller_batch',
            'description': 'Reduced batch size (8 vs 16)',
            'changes': {
                'IMS_PER_BATCH: 16': 'IMS_PER_BATCH: 8'
            }
        },
    ]
    
    configs = []
    
    for var in variations:
        config_content = base_content
        
        # Apply changes
        for old, new in var['changes'].items():
            config_content = config_content.replace(old, new)
        
        # Save config
        config_file = os.path.join(
            output_dir,
            f"config_{var['name']}.yml"
        )
        
        os.makedirs(output_dir, exist_ok=True)
        
        with open(config_file, 'w') as f:
            f.write(config_content)
        
        configs.append({
            'name': var['name'],
            'description': var['description'],
            'file': config_file,
        })
        
        print(f"✓ Created: {var['name']} - {var['description']}")
    
    return configs


def main():
    parser = argparse.ArgumentParser(
        description='Researcher tool for hyperparameter search and model selection'
    )
    parser.add_argument('--base-config', default='custom_configs/bagtricks_R50-ibn.yml',
                       help='Base configuration file')
    parser.add_argument('--output-dir', default='logs/research_experiment',
                       help='Output directory for all runs')
    parser.add_argument('--compare-only', action='store_true',
                       help='Only compare existing runs, do not train')
    parser.add_argument('--run-names', nargs='+',
                       help='Specific runs to train (space-separated)')
    
    args = parser.parse_args()
    
    experiment = ResearchExperiment(args.output_dir)
    
    print(f"\n{'='*70}")
    print("RESEARCHER HYPERPARAMETER SEARCH")
    print(f"{'='*70}\n")
    
    # Create config variations
    print("Creating configuration variations...")
    configs = create_search_configs(args.base_config, 
                                   os.path.join(args.output_dir, 'configs'))
    
    # Add runs to experiment
    for config in configs:
        experiment.add_run(
            config['name'],
            config['file'],
            {'description': config['description']}
        )
    
    # Run trainings (if not compare-only)
    if not args.compare_only:
        for run in experiment.runs:
            # Skip if specific runs requested and this isn't one
            if args.run_names and run['name'] not in args.run_names:
                print(f"⊘ Skipping: {run['name']}")
                continue
            
            success = experiment.run_training(
                run['name'],
                run['config']
            )
            
            if success:
                print(f"✓ Completed: {run['name']}")
            else:
                print(f"❌ Failed: {run['name']}")
    
    # Compare results
    comparison = experiment.compare_runs(args.output_dir)
    
    if comparison:
        experiment.save_best_model_report(comparison, args.output_dir)
    else:
        print("No results to compare!")


if __name__ == '__main__':
    main()
