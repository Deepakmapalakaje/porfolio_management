import pandas as pd
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# --- Configuration ---
TEMP_DATA_DIR = "temp_training_data"  # Directory where training data was saved
EXISTING_MODEL_PATH = "generalized_lstm_model.keras"  # Your best model from previous training
NEW_MODEL_PATH = "generalized_lstm_model_improved.keras"  # Save improved model with new name
ADDITIONAL_EPOCHS = 100  # Number of additional epochs to train
BATCH_SIZE = 512  # Same batch size as original training

# --- Data Generator (Same as original) ---
def get_dataset(file_list, data_dir, batch_size):
    """Create tf.data pipeline from saved .npz files"""
    def generator():
        # Shuffle file access order
        files = list(file_list)
        np.random.shuffle(files)
        for f in files:
            try:
                # Load file
                path = os.path.join(data_dir, f)
                with np.load(path) as data:
                    X = data['X']
                    y = data['y']
                
                # Yield the whole array; unbatch() will handle splitting
                yield X, y
            except:
                continue
    
    # Create dataset from generator
    ds = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=(None, 120, 31), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.float32)
        )
    )
    
    # Pipeline: Unbatch -> Shuffle -> Repeat -> Batch -> Prefetch
    ds = ds.unbatch()
    ds = ds.shuffle(buffer_size=50000)  # Large buffer for good mixing
    ds = ds.repeat()  # Repeat indefinitely to prevent "ran out of data" error
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    
    return ds

def main():
    print("="*70)
    print("  CONTINUE TRAINING - IMPROVE EXISTING MODEL")
    print("="*70)
    
    # 1. Check if existing model exists
    if not os.path.exists(EXISTING_MODEL_PATH):
        print(f"\n[ERROR] Model file '{EXISTING_MODEL_PATH}' not found!")
        print("Please ensure you have trained the model using train_generalized_model.py first.")
        return
    
    # 2. Check if training data exists
    if not os.path.exists(TEMP_DATA_DIR):
        print(f"\n[ERROR] Training data directory '{TEMP_DATA_DIR}' not found!")
        print("Please run train_generalized_model.py first to generate training data.")
        return
    
    # Get list of training files
    valid_files = [f for f in os.listdir(TEMP_DATA_DIR) if f.endswith('.npz')]
    
    if len(valid_files) == 0:
        print(f"\n[ERROR] No training data files found in '{TEMP_DATA_DIR}'!")
        print("Please run train_generalized_model.py first to generate training data.")
        return
    
    print(f"\n[SUCCESS] Found {len(valid_files)} training data files")
    
    # 3. Load the existing model
    print(f"\n[LOADING] Loading existing model from '{EXISTING_MODEL_PATH}'...")
    try:
        model = load_model(EXISTING_MODEL_PATH)
        print("[SUCCESS] Model loaded successfully!")
    except Exception as e:
        print(f"[ERROR] ERROR loading model: {e}")
        return
    
    # Display model summary
    print("\n[MODEL] Current Model Architecture:")
    model.summary()
    
    # 4. Prepare datasets
    print("\n[PREPARING] Preparing training and validation datasets...")
    
    # Split files for train/val (same split as original)
    np.random.shuffle(valid_files)
    split_idx = int(len(valid_files) * 0.8)
    train_files = valid_files[:split_idx]
    val_files = valid_files[split_idx:]
    
    print(f"  Training files: {len(train_files)}")
    print(f"  Validation files: {len(val_files)}")
    
    # Create tf.data pipelines
    print("\n[SETUP] Creating tf.data pipelines...")
    train_ds = get_dataset(train_files, TEMP_DATA_DIR, BATCH_SIZE)
    val_ds = get_dataset(val_files, TEMP_DATA_DIR, BATCH_SIZE)
    
    # Calculate steps per epoch
    print(f"\n[CALCULATING] Calculating dataset sizes...")
    total_train_samples = 0
    for f in train_files:
        try:
            d = np.load(os.path.join(TEMP_DATA_DIR, f))
            total_train_samples += d['X'].shape[0]
        except:
            pass
            
    total_val_samples = 0
    for f in val_files:
        try:
            d = np.load(os.path.join(TEMP_DATA_DIR, f))
            total_val_samples += d['X'].shape[0]
        except:
            pass
            
    steps_per_epoch = total_train_samples // BATCH_SIZE
    validation_steps = total_val_samples // BATCH_SIZE
    
    print(f"  Total Training Samples: {total_train_samples:,}")
    print(f"  Total Validation Samples: {total_val_samples:,}")
    print(f"  Steps per Epoch: {steps_per_epoch}")
    print(f"  Validation Steps: {validation_steps}")
    
    # 5. Setup callbacks for continued training
    print("\n[CALLBACKS] Setting up training callbacks...")
    
    # Save the improved model with a new name
    checkpoint = ModelCheckpoint(
        NEW_MODEL_PATH,
        monitor='val_loss',
        save_best_only=True,
        verbose=1,
        save_weights_only=False
    )
    
    # Learning Rate Scheduler - more aggressive for fine-tuning
    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,  # Reduced patience for faster adaptation
        min_lr=1e-7,  # Allow even lower learning rate
        verbose=1
    )
    
    # Early Stopping - patience for continued training
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=20,  # More patience since we're fine-tuning
        verbose=1,
        restore_best_weights=False,  # Best model saved by checkpoint
        min_delta=1e-7,  # Very sensitive for ultra-low loss tracking
        start_from_epoch=0  # Start monitoring immediately
    )
    
    # 6. Display training configuration
    print("\n" + "="*70)
    print("[CONFIG] CONTINUED TRAINING CONFIGURATION")
    print("="*70)
    print(f"  Additional Epochs:  {ADDITIONAL_EPOCHS}")
    print(f"  Early Stop:         Patience = 20 epochs")
    print(f"  LR Scheduler:       Reduce LR on plateau (patience=3)")
    print(f"  Batch Size:         {BATCH_SIZE}")
    print(f"  Current Optimizer:  {model.optimizer.__class__.__name__}")
    print(f"  Current LR:         {model.optimizer.learning_rate.numpy():.6f}")
    print(f"  Loss Function:      {model.loss}")
    print(f"  Improved Model:     Will be saved to '{NEW_MODEL_PATH}'")
    print("="*70)
    
    # Optional: Reduce learning rate for fine-tuning
    print("\n[TUNING] Adjusting learning rate for fine-tuning...")
    current_lr = model.optimizer.learning_rate.numpy()
    new_lr = current_lr * 0.5  # Reduce by half for more stable fine-tuning
    model.optimizer.learning_rate.assign(new_lr)
    print(f"  Learning rate reduced: {current_lr:.6f} -> {new_lr:.6f}")

    
    # 7. Continue training
    print("\n" + "="*70)
    print("[TRAINING] STARTING CONTINUED TRAINING...")
    print("="*70)
    print("\nThe model will continue learning from where it left off.")
    print("Training will stop early if no improvement is detected.\n")
    
    # Train the model
    history = model.fit(
        train_ds,
        epochs=ADDITIONAL_EPOCHS,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        validation_data=val_ds,
        callbacks=[checkpoint, lr_scheduler, early_stop],
        verbose=1
    )
    
    print("\n" + "="*70)
    print("[COMPLETE] CONTINUED TRAINING COMPLETE!")
    print("="*70)
    
    # 8. Save training history plot
    print("\n[PLOTTING] Generating training history plot...")
    
    plt.figure(figsize=(12, 6))
    
    # Plot training and validation loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss', linewidth=2)
    plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    plt.title('Continued Training: Loss vs Epochs', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss (Huber)', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Plot learning rate changes
    if 'lr' in history.history:
        plt.subplot(1, 2, 2)
        plt.plot(history.history['lr'], label='Learning Rate', color='orange', linewidth=2)
        plt.title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Learning Rate', fontsize=12)
        plt.yscale('log')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('continued_training_plot.png', dpi=150)
    print("[SUCCESS] Saved training plot to 'continued_training_plot.png'")
    
    # 9. Evaluate the improved model
    print("\n[EVALUATING] Evaluating improved model on validation set...")
    
    # Load the best improved model
    try:
        best_model = load_model(NEW_MODEL_PATH)
        print(f"[SUCCESS] Loaded best improved model from '{NEW_MODEL_PATH}'")
    except:
        print("[WARNING] Using current model for evaluation")
        best_model = model
    
    # Evaluate
    val_loss = best_model.evaluate(val_ds, steps=validation_steps, verbose=1)
    
    print("\n" + "="*70)
    print("[RESULTS] FINAL RESULTS")
    print("="*70)
    print(f"  Validation Loss: {val_loss:.8f}")
    print(f"  Best model saved to: '{NEW_MODEL_PATH}'")
    print(f"  Training plot saved to: 'continued_training_plot.png'")
    print("="*70)
    
    # 10. Compare with original model
    print("\n[COMPARING] Comparing with original model...")
    try:
        original_model = load_model(EXISTING_MODEL_PATH)
        original_val_loss = original_model.evaluate(val_ds, steps=validation_steps, verbose=0)
        
        print("\n" + "="*70)
        print("[COMPARISON] MODEL COMPARISON")
        print("="*70)
        print(f"  Original Model Loss:  {original_val_loss:.8f}")
        print(f"  Improved Model Loss:  {val_loss:.8f}")
        
        improvement = ((original_val_loss - val_loss) / original_val_loss) * 100
        if improvement > 0:
            print(f"  [SUCCESS] Improvement:        {improvement:.2f}% better!")
        else:
            print(f"  [INFO] Change:             {improvement:.2f}%")
            print("     (Model may have converged already)")
        print("="*70)
    except Exception as e:
        print(f"[WARNING] Could not compare models: {e}")
    
    print("\n[SUCCESS] All done! Your improved model is ready to use.")
    print(f"\n[TIP] Use '{NEW_MODEL_PATH}' for predictions in your analysis scripts.")

if __name__ == "__main__":
    # Set random seeds for reproducibility
    tf.random.set_seed(42)
    np.random.seed(42)
    
    # Run main function
    main()
