#!/usr/bin/env python
"""
FastReID Training Monitor with Real-time Progress Display
Shows training metrics with a beautiful progress bar
"""
import os
import sys
import subprocess
import time
import re
from datetime import datetime
from pathlib import Path

# ANSI Color codes for terminal
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_header():
    """Print fancy header"""
    print(f"\n{Colors.BOLD}{Colors.OKBLUE}")
    print("=" * 80)
    print("         🚀 FastReID Person Re-Identification Training 🚀")
    print("=" * 80)
    print(f"{Colors.ENDC}")

def print_config():
    """Print training configuration"""
    print(f"{Colors.OKGREEN}{Colors.BOLD}📋 Training Configuration:{Colors.ENDC}")
    print(f"  • Dataset: Market1501 (751 IDs, 12,936 images)")
    print(f"  • Model: ResNet50-IBN (Backbone with Instance Batch Normalization)")
    print(f"  • Losses: Cross-Entropy + Triplet Loss")
    print(f"  • Batch Size: 16 images")
    print(f"  • Max Epochs: 60")
    print(f"  • GPU: NVIDIA GeForce GTX 1650 Ti")
    print(f"  • Device: CUDA 11.8")
    print(f"  • Output Dir: logs/market_r50_ibn/")
    print()

def create_progress_bar(current, total, width=50, prefix=""):
    """Create a visual progress bar"""
    if total == 0:
        percent = 0
    else:
        percent = current / total
    
    filled = int(width * percent)
    bar = '█' * filled + '░' * (width - filled)
    
    # Color based on progress
    if percent < 0.33:
        color = Colors.FAIL
    elif percent < 0.66:
        color = Colors.WARNING
    else:
        color = Colors.OKGREEN
    
    percentage = f"{percent*100:.1f}%"
    return f"{color}[{bar}]{Colors.ENDC} {percentage:>6} {prefix}"

def monitor_training():
    """Monitor training and display progress"""
    # Set environment
    os.environ['FASTREID_DATASETS'] = 'fast-reid/datasets'
    os.environ['PYTHONPATH'] = os.path.join(os.getcwd(), 'fast-reid') + os.pathsep + os.environ.get('PYTHONPATH', '')
    
    log_file = Path('logs/market_r50_ibn/log.txt')
    
    # Build command
    cmd = [
        sys.executable,
        'fast-reid/tools/train_net.py',
        '--config-file', 'custom_configs/bagtricks_R50-ibn.yml',
        'OUTPUT_DIR', 'logs/market_r50_ibn'
    ]
    
    print_header()
    print_config()
    print(f"{Colors.BOLD}{Colors.OKBLUE}🔄 Starting Training...{Colors.ENDC}\n")
    
    # Start training process
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True
    )
    
    last_epoch = 0
    last_iter = 0
    last_loss = 0
    last_cls_loss = 0
    last_tri_loss = 0
    last_eta = ""
    update_count = 0
    
    print(f"{Colors.OKBLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{Colors.ENDC}")
    print(f"{Colors.BOLD}TRAINING PROGRESS{Colors.ENDC}")
    print(f"{Colors.OKBLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{Colors.ENDC}\n")
    
    # Pattern to match training log
    pattern = r'eta: ([\d\w:]+)\s+epoch/iter: (\d+)/(\d+)\s+total_loss: ([\d.]+)\s+loss_cls: ([\d.]+)\s+loss_triplet: ([\d.]+)'
    
    try:
        for line in process.stdout:
            # Check if it's a training log line
            match = re.search(pattern, line)
            if match:
                eta, epoch, iteration, total_loss, cls_loss, tri_loss = match.groups()
                epoch, iteration = int(epoch), int(iteration)
                total_loss, cls_loss, tri_loss = float(total_loss), float(cls_loss), float(tri_loss)
                
                last_epoch = epoch
                last_iter = iteration
                last_loss = total_loss
                last_cls_loss = cls_loss
                last_tri_loss = tri_loss
                last_eta = eta
                update_count += 1
                
                # Calculate iteration progress in current epoch
                # Assuming ~200 iterations per epoch (12936 images / 16 batch size ≈ 809 batches, but displayed every 100)
                iter_percent = (iteration % 200) / 200.0 if iteration > 0 else 0
                
                # Build progress display with proper line clearing
                progress_lines = []
                progress_lines.append(f"{Colors.BOLD}🎯 Epoch {last_epoch:02d}/{60}{Colors.ENDC} | " + 
                                    create_progress_bar(iteration % 200, 200, width=35, prefix=""))
                
                # Print losses with nice formatting
                progress_lines.append(f"  {Colors.WARNING}📊 Total Loss{Colors.ENDC}: {Colors.BOLD}{last_loss:.4f}{Colors.ENDC} | " +
                                    f"{Colors.OKCYAN}CE: {last_cls_loss:.4f}{Colors.ENDC} | " +
                                    f"{Colors.OKCYAN}Triplet: {last_tri_loss:.4f}{Colors.ENDC}")
                
                # Print ETA and iteration
                progress_lines.append(f"  {Colors.OKGREEN}⏱️  ETA: {last_eta}{Colors.ENDC} | " +
                                    f"{Colors.OKGREEN}Step: {iteration}{Colors.ENDC} | " +
                                    f"{Colors.OKGREEN}🔥 GPU Memory: ~3.1 GB{Colors.ENDC}")
                
                # Print all progress lines
                for line in progress_lines:
                    print(line)
                print()  # Empty line for spacing
                
            # Also print important messages
            elif 'Starting training' in line:
                print(f"{Colors.OKGREEN}{Colors.BOLD}✓ Training started! Loading model onto GPU...{Colors.ENDC}\n")
            elif 'Loading pretrained' in line:
                print(f"{Colors.OKBLUE}📥 Downloading pretrained ResNet50-IBN weights...{Colors.ENDC}")
            elif 'Prepare training' in line:
                print(f"{Colors.OKBLUE}📂 Preparing training dataset...{Colors.ENDC}")
    
    except KeyboardInterrupt:
        print(f"\n\n{Colors.WARNING}{Colors.BOLD}⚠️  Training interrupted by user (Ctrl+C){Colors.ENDC}")
        print(f"{Colors.WARNING}Gracefully shutting down...{Colors.ENDC}")
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
        sys.exit(1)
    
    # Wait for process to complete
    returncode = process.wait()
    
    print(f"\n{Colors.OKBLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{Colors.ENDC}")
    
    if returncode == 0:
        print(f"\n{Colors.OKGREEN}{Colors.BOLD}✅ Training completed successfully!{Colors.ENDC}")
        print(f"{Colors.OKGREEN}📊 Model saved to: logs/market_r50_ibn/{Colors.ENDC}")
        print(f"{Colors.OKGREEN}📈 Final Epoch: {last_epoch}/60{Colors.ENDC}")
        print(f"{Colors.OKGREEN}📉 Final Loss: {last_loss:.4f}{Colors.ENDC}\n")
    else:
        print(f"\n{Colors.FAIL}{Colors.BOLD}❌ Training stopped with return code {returncode}{Colors.ENDC}\n")
        sys.exit(1)

if __name__ == '__main__':
    monitor_training()
