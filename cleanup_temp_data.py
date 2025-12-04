"""
Cleanup script to remove temporary training data files.
Run this separately when you're sure no processes are using the files.
"""
import shutil
import os
import time

output_dir = "temp_training_data"

if os.path.exists(output_dir):
    print(f"Attempting to remove: {output_dir}")
    print("Please close any programs that might be using these files...")
    time.sleep(2)
    
    try:
        shutil.rmtree(output_dir)
        print(f"✓ Successfully removed {output_dir}")
    except PermissionError as e:
        print(f"✗ Error: {e}")
        print("\nTroubleshooting:")
        print("1. Close any Python processes or file explorers")
        print("2. Wait a few seconds and try again")
        print("3. Or manually delete the folder from File Explorer")
else:
    print(f"Directory '{output_dir}' does not exist.")
