"""
🚀 TURBO TRAINING SCRIPT - MAXIMUM SPEED OPTIMIZATION
=====================================================

This script runs the training with all speed optimizations enabled.
Estimated time reduction: 40-60% faster than original!

OPTIMIZATIONS APPLIED:
- ✅ Maximum CPU cores for parallel processing
- ✅ Larger batch sizes (16 vs 8)
- ✅ Reduced epochs (30 vs 50)
- ✅ Aggressive early stopping (patience=2)
- ✅ Faster data sampling (100 vs 200 sequences)
- ✅ Chunked parallel processing
- ✅ Less frequent progress updates

ESTIMATED TIMES:
- Data Generation: ~15-30 min (with max cores)
- Model Training: ~10-20 min (with early stopping)
- Total: ~25-50 min (vs 60-120 min original)

REQUIREMENTS:
- Multi-core CPU (more cores = faster)
- 8GB+ RAM recommended
- SSD for faster I/O

"""

import subprocess
import sys
import time
from datetime import datetime

def print_header():
    print("=" * 70)
    print("🚀 TURBO TRAINING MODE - MAXIMUM SPEED")
    print("=" * 70)
    print(f"⏰ Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    print()

def print_footer(start_time):
    elapsed = time.time() - start_time
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)
    print()
    print("=" * 70)
    print(f"✅ TRAINING COMPLETE!")
    print(f"⏱️  Total Time: {minutes}m {seconds}s")
    print(f"⏰ End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

def main():
    print_header()
    
    print("📋 Speed Optimizations:")
    print("  • Max CPU cores for parallel processing")
    print("  • Batch size: 16 (2x faster)")
    print("  • Epochs: 30 (40% reduction)")
    print("  • Early stopping: patience=2 (aggressive)")
    print("  • Data sampling: 100 sequences/file")
    print()
    
    input("Press ENTER to start turbo training... ")
    print()
    
    start_time = time.time()
    
    try:
        # Run the training script
        result = subprocess.run(
            [sys.executable, "train_generalized_model.py"],
            check=True
        )
        
        print_footer(start_time)
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Training failed with error code {e.returncode}")
        print(f"⏱️  Time before failure: {int((time.time() - start_time) // 60)}m")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        print(f"⏱️  Time elapsed: {int((time.time() - start_time) // 60)}m")
        sys.exit(1)

if __name__ == "__main__":
    main()
