#!/usr/bin/env python3
"""
Heart Segmentation Inference Script
===================================

Quick inference script to test the trained model on new images.
Provides easy verification of model performance.

Usage:
    python inference.py --input_path path/to/image.nii.gz --model_path models/best_model.h5
    
Requirements:
    - Trained model weights (best_model.h5)
    - Input NIfTI image
    - All dependencies from requirements.txt
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import tensorflow as tf
from pathlib import Path
import cv2

def load_and_preprocess_image(image_path, target_size=(128, 128)):
    """
    Load NIfTI image and preprocess for model inference
    
    Args:
        image_path: Path to .nii.gz file
        target_size: Target image dimensions
        
    Returns:
        Preprocessed image array ready for model input
    """
    try:
        # Load NIfTI image
        img = nib.load(image_path)
        img_data = img.get_fdata()
        
        # Extract middle slice
        middle_slice = img_data.shape[2] // 2
        slice_2d = img_data[:, :, middle_slice]
        
        # Normalize to [0, 1]
        slice_2d = (slice_2d - slice_2d.min()) / (slice_2d.max() - slice_2d.min())
        
        # Resize to target size
        slice_resized = cv2.resize(slice_2d, target_size, interpolation=cv2.INTER_LINEAR)
        
        # Add batch and channel dimensions
        slice_input = np.expand_dims(slice_resized, axis=[0, -1])
        
        return slice_input, slice_2d
        
    except Exception as e:
        print(f"Error loading image {image_path}: {str(e)}")
        return None, None

def create_demo_model():
    """
    Create a demo CustomCNN model architecture for inference
    This should match the trained model architecture
    """
    input_layer = tf.keras.layers.Input(shape=(128, 128, 1))
    
    # Encoder
    e1 = tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu')(input_layer)
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
    output = tf.keras.layers.Conv2D(1, 1, activation='sigmoid')(d2)
    
    model = tf.keras.Model(inputs=input_layer, outputs=output)
    return model

def visualize_prediction(original_slice, prediction, save_path=None):
    """
    Create visualization of input image and prediction
    
    Args:
        original_slice: Original 2D image slice
        prediction: Model prediction mask
        save_path: Optional path to save visualization
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(original_slice, cmap='gray')
    axes[0].set_title('Input Image')
    axes[0].axis('off')
    
    # Prediction mask
    pred_binary = (prediction.squeeze() > 0.5).astype(np.uint8)
    axes[1].imshow(pred_binary, cmap='gray')
    axes[1].set_title('Predicted Heart Mask')
    axes[1].axis('off')
    
    # Overlay
    axes[2].imshow(original_slice, cmap='gray', alpha=0.7)
    axes[2].imshow(pred_binary, cmap='Reds', alpha=0.5)
    axes[2].set_title('Overlay (Red = Predicted Heart)')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")
    
    plt.show()

def calculate_basic_metrics(prediction, threshold=0.5):
    """
    Calculate basic metrics for the prediction
    
    Args:
        prediction: Model output probabilities
        threshold: Binary threshold for prediction
        
    Returns:
        Dictionary of basic metrics
    """
    pred_binary = (prediction.squeeze() > threshold).astype(np.uint8) 
    
    # Basic statistics
    total_pixels = pred_binary.size
    heart_pixels = np.sum(pred_binary)
    heart_percentage = (heart_pixels / total_pixels) * 100
    
    # Prediction confidence statistics
    confidence_mean = np.mean(prediction)
    confidence_std = np.std(prediction)
    
    metrics = {
        'total_pixels': total_pixels,
        'predicted_heart_pixels': heart_pixels,
        'heart_percentage': heart_percentage,
        'confidence_mean': confidence_mean,
        'confidence_std': confidence_std,
        'prediction_range': (np.min(prediction), np.max(prediction))
    }
    
    return metrics

def main():
    parser = argparse.ArgumentParser(description='Heart Segmentation Inference')
    parser.add_argument('--input_path', type=str, required=True, 
                       help='Path to input NIfTI image')
    parser.add_argument('--model_path', type=str, default='models/best_model.h5',
                       help='Path to trained model weights')
    parser.add_argument('--output_dir', type=str, default='inference_results',
                       help='Directory to save results')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Binary threshold for segmentation')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print("🔍 Heart Segmentation Inference")
    print("================================")
    print(f"Input image: {args.input_path}")
    print(f"Model weights: {args.model_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Binary threshold: {args.threshold}")
    print()
    
    # Load and preprocess image
    print("📸 Loading and preprocessing image...")
    preprocessed_image, original_slice = load_and_preprocess_image(args.input_path)
    
    if preprocessed_image is None:
        print("❌ Failed to load image. Please check the file path and format.")
        return
    
    print(f"✅ Image loaded successfully. Shape: {preprocessed_image.shape}")
    
    # Load model
    print("🧠 Loading trained model...")
    try:
        if Path(args.model_path).exists():
            # Load model with weights
            model = create_demo_model()
            model.load_weights(args.model_path)
            print("✅ Model weights loaded successfully")
        else:
            print(f"❌ Model file not found: {args.model_path}")
            print("Please ensure you have trained the model and saved the weights.")
            return
            
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        return
    
    # Run inference
    print("🚀 Running inference...")
    prediction = model.predict(preprocessed_image, verbose=0)
    print("✅ Inference completed")
    
    # Calculate metrics
    metrics = calculate_basic_metrics(prediction, args.threshold)
    
    # Display results
    print("\n📊 Prediction Results:")
    print(f"   • Total pixels: {metrics['total_pixels']:,}")
    print(f"   • Predicted heart pixels: {metrics['predicted_heart_pixels']:,}")
    print(f"   • Heart region percentage: {metrics['heart_percentage']:.2f}%")  
    print(f"   • Prediction confidence (mean ± std): {metrics['confidence_mean']:.3f} ± {metrics['confidence_std']:.3f}")
    print(f"   • Prediction range: [{metrics['prediction_range'][0]:.3f}, {metrics['prediction_range'][1]:.3f}]")
    
    # Save visualization
    viz_path = output_dir / f"prediction_visualization.png"
    visualize_prediction(original_slice, prediction, save_path=viz_path)
    
    # Save prediction array
    pred_path = output_dir / "prediction_mask.npy"
    np.save(pred_path, prediction.squeeze())
    print(f"💾 Prediction mask saved to: {pred_path}")
    
    print(f"\n🎉 Inference completed! Results saved to: {args.output_dir}")

if __name__ == "__main__":
    main()
