import numpy as np
import os

# Change to the correct directory
os.chdir(r'c:\Users\Karthik M\Desktop\assignment')

# Load a sample file
data = np.load("temp_training_data/data_0.npz")

X = data['X']
y = data['y']

print("=" * 60)
print("TRAINING DATA FILE CONTENT")
print("=" * 60)
print(f"\nFeatures (X) shape: {X.shape}")
print(f"  - Number of training sequences: {X.shape[0]}")
print(f"  - Lookback period (days): {X.shape[1]}")
print(f"  - Features per day: {X.shape[2]}")
print(f"\nTarget (y) shape: {y.shape}")
print(f"  - Number of target values: {y.shape[0]}")
print(f"\nSample target values (return factors):")
print(f"  First: {y[0]:.4f}")
print(f"  Last: {y[-1]:.4f}")
print(f"  Mean: {y.mean():.4f}")
print(f"  Min: {y.min():.4f}")
print(f"  Max: {y.max():.4f}")
print("\n" + "=" * 60)
print("EXPLANATION:")
print("=" * 60)
print(f"YES - Each file contains {X.shape[1]} days of portfolio data")
print(f"Each sequence represents one sliding window of {X.shape[1]} days")
print(f"The target is the return factor {X.shape[1]//2} days later")
print("=" * 60)
