#!/usr/bin/env python3
"""
Single-file Colab training script for FastReID
Just run: !python colab_train.py
"""

import os
import sys
import shutil
import subprocess
import zipfile
from pathlib import Path

def setup_paths():
    """Setup working directory and Python paths"""
    # Get repo root
    repo_root = Path(__file__).parent.absolute()
    os.chdir(repo_root)
    
    # Add fast-reid to path
    fast_reid_path = repo_root / 'fast-reid'
    if str(fast_reid_path) not in sys.path:
        sys.path.insert(0, str(fast_reid_path))
    
    # Set environment
    os.environ['PYTHONPATH'] = str(fast_reid_path)
    os.environ['FASTREID_DATASETS'] = str(fast_reid_path / 'datasets')
    
    print(f"✓ Working directory: {os.getcwd()}")
    print(f"✓ Fast-Reid path: {fast_reid_path}")
    return repo_root

def check_dataset():
    """Check if dataset exists, copy from Drive if needed"""
    repo_root = Path(__file__).parent.absolute()
    dataset_dir = repo_root / 'fast-reid' / 'datasets' / 'Market-1501-v15.09.15'
    
    if dataset_dir.exists():
        print(f"✓ Dataset found: {dataset_dir}")
        return True
    
    print("⚠ Dataset not found locally, checking Drive...")
    
    # Try to copy from Drive
    drive_zip = Path('/content/drive/MyDrive/reid training/Market-1501-v15.09.15.zip')
    drive_folder = Path('/content/drive/MyDrive/reid training/Market-1501-v15.09.15')
    
    if drive_zip.exists():
        print(f"Extracting dataset from {drive_zip}...")
        dataset_dir.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(drive_zip, 'r') as zip_ref:
            zip_ref.extractall(dataset_dir.parent)
        print(f"✓ Dataset extracted to {dataset_dir}")
        return True
    
    elif drive_folder.exists():
        print(f"Copying dataset from {drive_folder}...")
        dataset_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(drive_folder, dataset_dir)
        print(f"✓ Dataset copied to {dataset_dir}")
        return True
    
    else:
        print("✗ Dataset not found in Drive either!")
        print("  Upload Market-1501-v15.09.15.zip to: /My Drive/reid training/")
        return False

def install_dependencies():
    """Install required packages"""
    print("\nInstalling dependencies...")
    
    packages = [
        'opencv-python',
        'faiss-cpu',
        'yacs',
        'termcolor',
        'tabulate',
        'cloudpickle',
        'tqdm',
        'wheel',
        'scikit-learn',
        'tensorboard'
    ]
    
    subprocess.run(
        ['pip', 'install', '-q'] + packages,
        check=False  # Don't fail on warning
    )
    print("✓ Dependencies installed")

def verify_imports():
    """Verify all required modules can be imported"""
    print("\nVerifying imports...")
    
    required = {
        'torch': 'torch',
        'cv2': 'opencv-python',
        'yacs': 'yacs',
        'faiss': 'faiss-cpu',
        'sklearn': 'scikit-learn',
        'fastreid': 'fastreid (from fast-reid)'
    }
    
    missing = []
    for module, package in required.items():
        try:
            __import__(module)
            print(f"  ✓ {module}")
        except ImportError:
            print(f"  ✗ {module} (from {package})")
            missing.append(package)
    
    if missing:
        print(f"\n⚠ Missing: {', '.join(missing)}")
        return False
    
    print("✓ All imports successful")
    return True

def train_model(config_file=None, epochs=1, eval_period=1, run_name='colab_run'):
    """Train the model"""
    
    if config_file is None:
        config_file = 'custom_configs/bagtricks_R50-ibn.yml'
    
    print(f"\n{'='*60}")
    print(f"Starting Training")
    print(f"{'='*60}")
    print(f"Config: {config_file}")
    print(f"Epochs: {epochs}")
    print(f"Eval Period: {eval_period}")
    print(f"Run Name: {run_name}")
    print(f"{'='*60}\n")
    
    cmd = [
        'python', 'train_research_grade.py',
        '--config-file', config_file,
        '--run-name', run_name,
        'SOLVER.MAX_EPOCHS', str(epochs),
        'TEST.EVAL_PERIOD', str(eval_period)
    ]
    
    result = subprocess.run(cmd, cwd=os.getcwd())
    
    if result.returncode == 0:
        print(f"\n✓ Training completed successfully!")
        print(f"Models saved to: logs/market1501/{run_name}/")
        return True
    else:
        print(f"\n✗ Training failed with error code {result.returncode}")
        return False

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Colab FastReID Training')
    parser.add_argument('--config', default='custom_configs/bagtricks_R50-ibn.yml',
                       help='Config file path')
    parser.add_argument('--epochs', type=int, default=1,
                       help='Number of epochs (default: 1 for test)')
    parser.add_argument('--eval-period', type=int, default=1,
                       help='Evaluation period (default: 1 for test)')
    parser.add_argument('--run-name', default='colab_run',
                       help='Run name for logging')
    parser.add_argument('--full', action='store_true',
                       help='Full training (120 epochs, Solution 1)')
    
    args = parser.parse_args()
    
    # Override for full training
    if args.full:
        args.config = 'custom_configs/plateau_solutions/solution_1_higher_lr.yml'
        args.epochs = 90
        args.eval_period = 10
        args.run_name = 'solution_1_full'
    
    print(f"FastReID Colab Training v1.0\n")
    
    # Setup
    setup_paths()
    install_dependencies()
    
    if not check_dataset():
        print("\n✗ Cannot proceed without dataset")
        sys.exit(1)
    
    if not verify_imports():
        print("\n⚠ Proceeding anyway - some imports may fail")
    
    # Train
    success = train_model(
        config_file=args.config,
        epochs=args.epochs,
        eval_period=args.eval_period,
        run_name=args.run_name
    )
    
    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main()
