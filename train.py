#!/usr/bin/env python3
"""
Heart Segmentation Training Script
=================================

Standalone training script for heart segmentation models.
Can be used to reproduce all experimental results.

Usage:
    python train.py --experiment dropout --epochs 20 --batch_size 4
    python train.py --experiment extra_conv --epochs 20 --batch_size 4
    python train.py --experiment all  # Run all experiments
    
Requirements:
    - NIfTI heart dataset in HEART_DATASET folder
    - All dependencies from requirements.txt
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import tensorflow as tf
from pathlib import Path
import json

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

def load_dataset(data_dir, num_train=10, num_test=5, image_size=128):
    """
    Load and preprocess the heart segmentation dataset
    
    Args:
        data_dir: Path to HEART_DATASET directory
        num_train: Number of training samples to use
        num_test: Number of test samples to use  
        image_size: Target image size (square)
        
    Returns:
        Tuple of (train_images, train_masks, test_images, test_masks)
    """
    data_path = Path(data_dir)
    
    # Get file paths
    train_img_files = sorted(list((data_path / "images" / "training").glob("*.nii*")))[:num_train]
    train_mask_files = sorted(list((data_path / "segmentations" / "training").glob("*.nii*")))[:num_train]
    test_img_files = sorted(list((data_path / "images" / "testing").glob("*.nii*")))[:num_test]
    test_mask_files = sorted(list((data_path / "segmentations" / "testing").glob("*.nii*")))[:num_test]
    
    def process_files(img_files, mask_files):
        images, masks = [], []
        for img_file, mask_file in zip(img_files, mask_files):
            # Load NIfTI files
            img = nib.load(img_file).get_fdata()
            mask = nib.load(mask_file).get_fdata()
            
            # Extract middle slice
            mid_slice = img.shape[2] // 2
            img_slice = img[:, :, mid_slice]
            mask_slice = mask[:, :, mid_slice]
            
            # Normalize image
            img_slice = (img_slice - img_slice.min()) / (img_slice.max() - img_slice.min())
            
            # Resize
            img_resized = tf.image.resize(img_slice[..., None], [image_size, image_size]).numpy()
            mask_resized = tf.image.resize(mask_slice[..., None], [image_size, image_size], 
                                         method='nearest').numpy()
            
            # Binarize mask
            mask_resized = (mask_resized > 0.5).astype(np.float32)
            
            images.append(img_resized)
            masks.append(mask_resized)
            
        return np.array(images), np.array(masks)
    
    print(f"Loading {len(train_img_files)} training and {len(test_img_files)} test samples...")
    train_images, train_masks = process_files(train_img_files, train_mask_files)
    test_images, test_masks = process_files(test_img_files, test_mask_files)
    
    print(f"Dataset loaded: Train {train_images.shape}, Test {test_images.shape}")
    return train_images, train_masks, test_images, test_masks

def create_base_model(input_shape=(128, 128, 1)):
    """Create base CustomCNN model"""
    inputs = tf.keras.Input(shape=input_shape)
    
    # Encoder
    e1 = tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu')(inputs)
    e1 = tf.keras.layers.BatchNormalization()(e1)
    e1_pool = tf.keras.layers.MaxPooling2D(2)(e1)
    
    e2 = tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu')(e1_pool)
    e2 = tf.keras.layers.BatchNormalization()(e2)
    e2_pool = tf.keras.layers.MaxPooling2D(2)(e2)
    
    # Bottleneck
    bottleneck = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu')(e2_pool)
    bottleneck = tf.keras.layers.BatchNormalization()(bottleneck)
    
    # Decoder
    d1 = tf.keras.layers.UpSampling2D(2)(bottleneck)
    d1 = tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu')(d1)
    d1 = tf.keras.layers.BatchNormalization()(d1)
    
    d2 = tf.keras.layers.UpSampling2D(2)(d1)
    d2 = tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu')(d2)
    d2 = tf.keras.layers.BatchNormalization()(d2)
    
    # Output
    outputs = tf.keras.layers.Conv2D(1, 1, activation='sigmoid')(d2)
    
    return tf.keras.Model(inputs, outputs, name="CustomCNN_Base")

def create_dropout_model(input_shape=(128, 128, 1)):
    """Create CustomCNN with dropout regularization"""
    inputs = tf.keras.Input(shape=input_shape)
    
    # Encoder
    e1 = tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu')(inputs)
    e1 = tf.keras.layers.BatchNormalization()(e1)
    e1_pool = tf.keras.layers.MaxPooling2D(2)(e1)
    
    e2 = tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu')(e1_pool)
    e2 = tf.keras.layers.BatchNormalization()(e2)  
    e2_pool = tf.keras.layers.MaxPooling2D(2)(e2)
    
    # Bottleneck
    bottleneck = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu')(e2_pool)
    bottleneck = tf.keras.layers.BatchNormalization()(bottleneck)
    
    # Decoder with dropout
    d1 = tf.keras.layers.UpSampling2D(2)(bottleneck)
    d1 = tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu')(d1)
    d1 = tf.keras.layers.BatchNormalization()(d1)
    d1 = tf.keras.layers.Dropout(0.3)(d1)  # Key addition
    
    d2 = tf.keras.layers.UpSampling2D(2)(d1)
    d2 = tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu')(d2)
    d2 = tf.keras.layers.BatchNormalization()(d2)
    
    # Output
    outputs = tf.keras.layers.Conv2D(1, 1, activation='sigmoid')(d2)
    
    return tf.keras.Model(inputs, outputs, name="CustomCNN_Dropout")

def create_extra_conv_model(input_shape=(128, 128, 1)):
    """Create CustomCNN with extra convolution layer"""
    inputs = tf.keras.Input(shape=input_shape)
    
    # Encoder
    e1 = tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu')(inputs)
    e1 = tf.keras.layers.BatchNormalization()(e1)
    e1_pool = tf.keras.layers.MaxPooling2D(2)(e1)
    
    e2 = tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu')(e1_pool)
    e2 = tf.keras.layers.BatchNormalization()(e2)
    e2_pool = tf.keras.layers.MaxPooling2D(2)(e2)
    
    # Expanded bottleneck  
    bottleneck = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu')(e2_pool)
    bottleneck = tf.keras.layers.BatchNormalization()(bottleneck)
    bottleneck = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu')(bottleneck)  # Extra layer
    bottleneck = tf.keras.layers.BatchNormalization()(bottleneck)
    
    # Decoder
    d1 = tf.keras.layers.UpSampling2D(2)(bottleneck)
    d1 = tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu')(d1)
    d1 = tf.keras.layers.BatchNormalization()(d1)
    
    d2 = tf.keras.layers.UpSampling2D(2)(d1)
    d2 = tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu')(d2)
    d2 = tf.keras.layers.BatchNormalization()(d2)
    
    # Output
    outputs = tf.keras.layers.Conv2D(1, 1, activation='sigmoid')(d2)
    
    return tf.keras.Model(inputs, outputs, name="CustomCNN_ExtraConv")

def dice_coefficient(y_true, y_pred, smooth=1e-6):
    """Dice coefficient metric"""
    y_true_f = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
    y_pred_f = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

def iou_score(y_true, y_pred, smooth=1e-6):
    """IoU metric"""
    y_true_f = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
    y_pred_f = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)

def train_model(model, train_images, train_masks, test_images, test_masks, 
                epochs=20, batch_size=4, learning_rate=1e-3, experiment_name="base"):
    """
    Train a heart segmentation model
    
    Args:
        model: Keras model to train
        train_images, train_masks: Training data
        test_images, test_masks: Test data  
        epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate for optimizer
        experiment_name: Name for saving results
        
    Returns:
        Trained model and training history
    """
    print(f"\n🚀 Training {experiment_name} model...")
    print(f"   • Architecture: {model.name}")
    print(f"   • Parameters: {model.count_params():,}")
    print(f"   • Epochs: {epochs}")
    print(f"   • Batch size: {batch_size}")
    print(f"   • Learning rate: {learning_rate}")
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=[dice_coefficient, iou_score, 'accuracy']
    )
    
    # Callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            f'models/{experiment_name}_best.h5',
            monitor='val_dice_coefficient', 
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
    ]
    
    # Train model
    history = model.fit(
        train_images, train_masks,
        validation_data=(test_images, test_masks),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate final performance
    train_metrics = model.evaluate(train_images, train_masks, verbose=0)
    test_metrics = model.evaluate(test_images, test_masks, verbose=0)
    
    results = {
        'experiment': experiment_name,
        'model_name': model.name,
        'parameters': model.count_params(),
        'epochs_trained': len(history.history['loss']),
        'train_loss': float(train_metrics[0]),
        'train_dice': float(train_metrics[1]),
        'train_iou': float(train_metrics[2]),
        'train_accuracy': float(train_metrics[3]),
        'val_loss': float(test_metrics[0]),
        'val_dice': float(test_metrics[1]),
        'val_iou': float(test_metrics[2]),
        'val_accuracy': float(test_metrics[3]),
        'history': {k: [float(x) for x in v] for k, v in history.history.items()}
    }
    
    # Save results
    with open(f'results/{experiment_name}_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✅ Training completed!")
    print(f"   • Final Dice: {results['val_dice']:.4f}")
    print(f"   • Final IoU: {results['val_iou']:.4f}")
    print(f"   • Validation Loss: {results['val_loss']:.4f}")
    
    return model, history, results

def plot_training_curves(history, experiment_name):
    """Plot and save training curves"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Loss
    axes[0,0].plot(history.history['loss'], label='Training Loss')
    axes[0,0].plot(history.history['val_loss'], label='Validation Loss')
    axes[0,0].set_title('Model Loss')
    axes[0,0].set_xlabel('Epoch')
    axes[0,0].set_ylabel('Loss')
    axes[0,0].legend()
    
    # Dice coefficient
    axes[0,1].plot(history.history['dice_coefficient'], label='Training Dice')
    axes[0,1].plot(history.history['val_dice_coefficient'], label='Validation Dice')
    axes[0,1].set_title('Dice Coefficient')
    axes[0,1].set_xlabel('Epoch')
    axes[0,1].set_ylabel('Dice')
    axes[0,1].legend()
    
    # IoU
    axes[1,0].plot(history.history['iou_score'], label='Training IoU')
    axes[1,0].plot(history.history['val_iou_score'], label='Validation IoU')
    axes[1,0].set_title('IoU Score')
    axes[1,0].set_xlabel('Epoch')
    axes[1,0].set_ylabel('IoU')
    axes[1,0].legend()
    
    # Accuracy
    axes[1,1].plot(history.history['accuracy'], label='Training Accuracy')
    axes[1,1].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axes[1,1].set_title('Accuracy')
    axes[1,1].set_xlabel('Epoch')
    axes[1,1].set_ylabel('Accuracy')
    axes[1,1].legend()
    
    plt.tight_layout()
    plt.savefig(f'results/{experiment_name}_training_curves.png', dpi=150)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Heart Segmentation Training')
    parser.add_argument('--experiment', type=str, default='dropout',
                       choices=['base', 'dropout', 'extra_conv', 'all'],
                       help='Experiment to run')
    parser.add_argument('--data_dir', type=str, default='HEART_DATASET',
                       help='Path to dataset directory')
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--image_size', type=int, default=128,
                       help='Input image size (square)')
    
    args = parser.parse_args()
    
    # Create directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    print("🔬 Heart Segmentation Training")
    print("==============================")
    print(f"Experiment: {args.experiment}")
    print(f"Dataset: {args.data_dir}")
    print(f"Image size: {args.image_size}×{args.image_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    
    # Load dataset  
    train_images, train_masks, test_images, test_masks = load_dataset(
        args.data_dir, image_size=args.image_size
    )
    
    # Define experiments
    experiments = {
        'base': create_base_model,
        'dropout': create_dropout_model, 
        'extra_conv': create_extra_conv_model
    }
    
    if args.experiment == 'all':
        experiment_list = ['base', 'dropout', 'extra_conv']
    else:
        experiment_list = [args.experiment]
    
    # Run experiments
    all_results = []
    for exp_name in experiment_list:
        print(f"\n{'='*50}")
        print(f"Running experiment: {exp_name}")
        print(f"{'='*50}")
        
        # Create model
        model = experiments[exp_name]((args.image_size, args.image_size, 1))
        
        # Train model
        trained_model, history, results = train_model(
            model, train_images, train_masks, test_images, test_masks,
            epochs=args.epochs, batch_size=args.batch_size, 
            learning_rate=args.learning_rate, experiment_name=exp_name
        )
        
        # Plot training curves
        plot_training_curves(history, exp_name)
        
        all_results.append(results)
    
    # Save summary results
    with open('results/all_experiments_summary.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Print summary
    print(f"\n🎉 All experiments completed!")
    print("\n📊 Results Summary:")
    print("-" * 80)
    print(f"{'Experiment':<15} {'Dice':<8} {'IoU':<8} {'Val Loss':<10} {'Parameters':<12}")
    print("-" * 80)
    for result in all_results:
        print(f"{result['experiment']:<15} {result['val_dice']:<8.4f} "
              f"{result['val_iou']:<8.4f} {result['val_loss']:<10.4f} "
              f"{result['parameters']:<12,}")
    print("-" * 80)

if __name__ == "__main__":
    main()
