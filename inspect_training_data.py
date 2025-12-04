import numpy as np
import os

# Load a sample file
data_file = "temp_training_data/data_0.npz"

if os.path.exists(data_file):
    data = np.load(data_file)
    
    print("=" * 60)
    print("TRAINING DATA FILE INSPECTION")
    print("=" * 60)
    print(f"\nFile: {data_file}")
    print(f"\nKeys in file: {list(data.keys())}")
    
    X = data['X']
    y = data['y']
    
    print(f"\n--- Features (X) ---")
    print(f"Shape: {X.shape}")
    print(f"  - Number of sequences: {X.shape[0]}")
    print(f"  - Lookback period (days): {X.shape[1]}")
    print(f"  - Number of features: {X.shape[2]}")
    print(f"Data type: {X.dtype}")
    print(f"Memory size: {X.nbytes / 1024:.2f} KB")
    
    print(f"\n--- Target (y) ---")
    print(f"Shape: {y.shape}")
    print(f"  - Number of target values: {y.shape[0]}")
    print(f"Data type: {y.dtype}")
    print(f"Memory size: {y.nbytes / 1024:.2f} KB")
    
    print(f"\n--- Sample Data ---")
    print(f"First target value (return factor): {y[0]:.4f}")
    print(f"Last target value (return factor): {y[-1]:.4f}")
    
    print(f"\n--- Interpretation ---")
    print(f"This file contains {X.shape[0]} training examples.")
    print(f"Each example uses {X.shape[1]} days of historical data (lookback).")
    print(f"Each day has {X.shape[2]} technical indicators/features.")
    print(f"The target is the return factor after 60 days.")
    
    # Check a few more files
    print(f"\n--- Checking Multiple Files ---")
    for i in [0, 100, 500, 1000, 2000, 4000]:
        file_path = f"temp_training_data/data_{i}.npz"
        if os.path.exists(file_path):
            d = np.load(file_path)
            print(f"data_{i}.npz: {d['X'].shape[0]} sequences, {d['X'].shape[1]} days lookback")
    
    print("\n" + "=" * 60)
else:
    print(f"File not found: {data_file}")
