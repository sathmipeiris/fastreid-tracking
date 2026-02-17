#!/usr/bin/env python
# encoding: utf-8
"""
Visualization and analysis script for training with early stopping.
Plots validation metrics, detects overfitting, and provides training insights.
"""

import json
import argparse
import os
from pathlib import Path

def analyze_validation_history(history_file, output_dir=None):
    """
    Analyze validation history and detect overfitting patterns.
    
    Args:
        history_file: Path to validation_history.json
        output_dir: Optional directory to save plots
    """
    
    if not os.path.exists(history_file):
        print(f"ERROR: History file not found: {history_file}")
        return None
    
    with open(history_file, 'r') as f:
        history = json.load(f)
    
    print("\n" + "="*70)
    print("VALIDATION HISTORY ANALYSIS")
    print("="*70)
    
    epochs = history['epoch']
    mAPs = history['mAP']
    top1s = history['top1']
    train_losses = history['train_loss']
    overfitting_flags = history.get('overfitting_flag', [False] * len(epochs))
    
    if not epochs:
        print("No validation history found!")
        return None
    
    # Summary statistics
    print(f"\nTraining Duration: {len(epochs)} epochs")
    print(f"Best mAP: {max(mAPs):.4f} at epoch {epochs[mAPs.index(max(mAPs))]}")
    print(f"Best top-1: {max(top1s):.4f} at epoch {epochs[top1s.index(max(top1s))]}")
    print(f"Final mAP: {mAPs[-1]:.4f}")
    print(f"Final top-1: {top1s[-1]:.4f}")
    
    # Overfitting detection
    overfitting_epochs = [epochs[i] for i, flag in enumerate(overfitting_flags) if flag]
    if overfitting_epochs:
        print(f"\n⚠ OVERFITTING DETECTED at epochs: {overfitting_epochs}")
    else:
        print(f"\n✓ No overfitting detected")
    
    # Trends analysis
    print("\nMetric Trends:")
    
    # mAP trend
    mAP_improvement = mAPs[-1] - mAPs[0]
    print(f"  mAP improvement: {mAP_improvement:+.4f} ({mAP_improvement/mAPs[0]*100:+.1f}%)")
    
    # Check if mAP is still improving
    last_5_mAPs = mAPs[-5:] if len(mAPs) >= 5 else mAPs
    is_improving = all(last_5_mAPs[i] <= last_5_mAPs[i+1] for i in range(len(last_5_mAPs)-1))
    if is_improving:
        print(f"  Status: Still improving in last 5 epochs ✓")
    else:
        print(f"  Status: Plateaued or declining in last 5 epochs")
    
    # Variance analysis (stability)
    mAP_variance = sum((x - sum(mAPs)/len(mAPs))**2 for x in mAPs) / len(mAPs)
    print(f"  mAP variance: {mAP_variance:.6f} (lower = more stable)")
    
    # Training loss analysis
    loss_improvement = train_losses[0] - train_losses[-1]
    print(f"\nTraining Loss:")
    print(f"  Initial: {train_losses[0]:.4f}")
    print(f"  Final: {train_losses[-1]:.4f}")
    print(f"  Improvement: {loss_improvement:+.4f}")
    
    # Gap analysis (training loss vs validation performance)
    print(f"\nOverfitting Gap Analysis:")
    avg_mAP = sum(mAPs) / len(mAPs)
    print(f"  Average mAP: {avg_mAP:.4f}")
    print(f"  Best mAP: {max(mAPs):.4f}")
    print(f"  Performance range: {max(mAPs) - min(mAPs):.4f}")
    
    # Detailed epoch-by-epoch analysis
    print("\n" + "-"*70)
    print("EPOCH-BY-EPOCH BREAKDOWN")
    print("-"*70)
    print(f"{'Epoch':<8} {'mAP':<10} {'top-1':<10} {'Loss':<10} {'Status':<20}")
    print("-"*70)
    
    for i, epoch in enumerate(epochs):
        status = "⚠ OVERFITTING" if overfitting_flags[i] else "✓ OK"
        
        # Add trend indicator
        if i > 0:
            if mAPs[i] > mAPs[i-1]:
                trend = "↑"
            elif mAPs[i] < mAPs[i-1]:
                trend = "↓"
            else:
                trend = "→"
        else:
            trend = ""
        
        print(f"{epoch:<8} {mAPs[i]:.4f} {trend:<9} {top1s[i]:.4f}   {train_losses[i]:.4f}   {status:<20}")
    
    print("-"*70)
    
    # Recommendations
    print("\nRECOMMENDATIONS:")
    
    if max(mAPs) < 0.3:
        print("⚠ Low mAP overall. Consider:")
        print("  - Training longer (increase MAX_EPOCH)")
        print("  - Tuning learning rate")
        print("  - Using more diverse training data")
    
    if is_improving:
        print("✓ Model is still improving. Training completed optimally.")
    else:
        print("ℹ Model has plateaued. Early stopping prevented overfitting.")
    
    if len(overfitting_epochs) > 0:
        print("⚠ Overfitting detected. Consider:")
        print("  - Adding regularization (dropout, weight decay)")
        print("  - Using data augmentation")
        print("  - Reducing model complexity")
    
    print("="*70 + "\n")
    
    return history


def plot_validation_history(history_file, output_file=None):
    """
    Create plots of validation history.
    Requires matplotlib.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed. Skipping plot generation.")
        print("Install with: pip install matplotlib")
        return
    
    with open(history_file, 'r') as f:
        history = json.load(f)
    
    epochs = history['epoch']
    mAPs = history['mAP']
    top1s = history['top1']
    train_losses = history['train_loss']
    overfitting_flags = history.get('overfitting_flag', [False] * len(epochs))
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('ReID Training with Early Stopping', fontsize=16, fontweight='bold')
    
    # Plot 1: mAP over epochs
    ax = axes[0, 0]
    ax.plot(epochs, mAPs, 'b-o', linewidth=2, markersize=6, label='mAP')
    ax.axhline(y=max(mAPs), color='g', linestyle='--', alpha=0.5, label=f'Best mAP: {max(mAPs):.4f}')
    
    # Highlight overfitting epochs
    overfitting_epochs = [epochs[i] for i, flag in enumerate(overfitting_flags) if flag]
    if overfitting_epochs:
        overfitting_mAPs = [mAPs[i] for i, flag in enumerate(overfitting_flags) if flag]
        ax.scatter(overfitting_epochs, overfitting_mAPs, color='red', s=100, marker='X', 
                  label='Overfitting', zorder=5)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('mAP')
    ax.set_title('Mean Average Precision (mAP)')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Plot 2: Top-1 accuracy over epochs
    ax = axes[0, 1]
    ax.plot(epochs, top1s, 'g-s', linewidth=2, markersize=6)
    ax.axhline(y=max(top1s), color='g', linestyle='--', alpha=0.5, label=f'Best: {max(top1s):.4f}')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Top-1 Accuracy')
    ax.set_title('Top-1 Accuracy')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Plot 3: Training loss over epochs
    ax = axes[1, 0]
    ax.plot(epochs, train_losses, 'r-^', linewidth=2, markersize=6)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss')
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Combined view (normalized)
    ax = axes[1, 1]
    if max(mAPs) > 0:
        norm_mAP = [m / max(mAPs) for m in mAPs]
    else:
        norm_mAP = mAPs
    
    if max(top1s) > 0:
        norm_top1 = [t / max(top1s) for t in top1s]
    else:
        norm_top1 = top1s
    
    if max(train_losses) > 0:
        norm_loss = [1 - (l / max(train_losses)) for l in train_losses]  # Inverted so higher is better
    else:
        norm_loss = train_losses
    
    ax.plot(epochs, norm_mAP, 'b-o', label='mAP (normalized)', linewidth=2)
    ax.plot(epochs, norm_top1, 'g-s', label='Top-1 (normalized)', linewidth=2)
    ax.plot(epochs, norm_loss, 'r-^', label='1 - Loss (normalized)', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Normalized Value')
    ax.set_title('All Metrics (Normalized)')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_ylim([0, 1.1])
    
    plt.tight_layout()
    
    # Save plot
    if output_file is None:
        output_file = str(Path(history_file).parent / 'validation_plots.png')
    
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {output_file}")
    
    try:
        plt.show()
    except:
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze validation history from early stopping training')
    parser.add_argument('history_file', help='Path to validation_history.json file')
    parser.add_argument('--output', '-o', help='Output file for plots (optional)')
    parser.add_argument('--plot', '-p', action='store_true', help='Generate plots')
    
    args = parser.parse_args()
    
    # Analyze
    history = analyze_validation_history(args.history_file)
    
    # Plot if requested
    if args.plot and history:
        plot_validation_history(args.history_file, args.output)
