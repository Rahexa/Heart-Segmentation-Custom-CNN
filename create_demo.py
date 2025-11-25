#!/usr/bin/env python3
"""
Create demo sample input and output for quick verification
This script generates sample data that can be used to test the trained model
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json

def create_demo_sample():
    """Create a synthetic heart-like sample for demo purposes"""
    # Create a 128x128 synthetic image resembling a chest X-ray
    image = np.zeros((128, 128))
    
    # Add background noise
    np.random.seed(42)
    image += np.random.normal(0.1, 0.05, (128, 128))
    
    # Create heart-like structures
    y, x = np.ogrid[:128, :128]
    
    # Main heart body (left ventricle area)
    heart_center_x, heart_center_y = 64, 70
    heart1 = ((x - heart_center_x) ** 2 / 20**2 + (y - heart_center_y) ** 2 / 15**2) < 1
    
    # Secondary heart chamber (right ventricle area)  
    heart_center_x2, heart_center_y2 = 45, 65
    heart2 = ((x - heart_center_x2) ** 2 / 12**2 + (y - heart_center_y2) ** 2 / 10**2) < 1
    
    # Combine heart regions
    heart_mask = heart1 | heart2
    
    # Add heart regions to image with higher intensity
    image[heart_mask] = 0.8 + np.random.normal(0, 0.1, np.sum(heart_mask))
    
    # Add some rib-like structures
    for i in range(3):
        rib_y = 30 + i * 20
        rib_mask = (np.abs(y - rib_y) < 2) & (x > 20) & (x < 100)
        image[rib_mask] = 0.6
    
    # Normalize to [0, 1]
    image = np.clip(image, 0, 1)
    
    # Create ground truth mask (binary)
    ground_truth = heart_mask.astype(np.float32)
    
    return image, ground_truth

def create_expected_prediction():
    """Create expected model prediction for the demo sample"""
    # This simulates what a trained model might predict
    np.random.seed(42)
    
    # Start with the ground truth shape but add realistic imperfections
    _, gt_mask = create_demo_sample()
    
    # Add some prediction uncertainty/noise
    prediction = gt_mask.astype(np.float32)
    
    # Add some false positives (small regions wrongly predicted as heart)
    noise_mask = np.random.random((128, 128)) > 0.98
    prediction[noise_mask] = 0.7
    
    # Add some false negatives (remove some true heart regions)
    remove_mask = np.random.random((128, 128)) > 0.95
    prediction[remove_mask & (prediction > 0)] = 0.1
    
    # Smooth the prediction to make it more realistic
    from scipy import ndimage
    prediction = ndimage.gaussian_filter(prediction, sigma=0.5)
    
    # Ensure values are in [0, 1] range (as sigmoid output)
    prediction = np.clip(prediction, 0, 1)
    
    return prediction

def create_demo_visualization():
    """Create a comprehensive demo visualization"""
    # Generate demo data
    input_image, ground_truth = create_demo_sample()
    predicted_mask = create_expected_prediction()
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Row 1: Input and masks
    axes[0, 0].imshow(input_image, cmap='gray')
    axes[0, 0].set_title('Demo Input Image\n(Synthetic Chest X-ray)')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(ground_truth, cmap='gray')
    axes[0, 1].set_title('Ground Truth Mask\n(Expert Annotation)')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(predicted_mask, cmap='viridis')
    axes[0, 2].set_title('Model Prediction\n(Probability Map)')
    axes[0, 2].axis('off')
    
    # Row 2: Overlays and binary prediction
    axes[1, 0].imshow(input_image, cmap='gray', alpha=0.7)
    axes[1, 0].imshow(ground_truth, cmap='Reds', alpha=0.5)
    axes[1, 0].set_title('Ground Truth Overlay\n(Red = True Heart)')
    axes[1, 0].axis('off')
    
    binary_pred = (predicted_mask > 0.5).astype(np.float32)
    axes[1, 1].imshow(binary_pred, cmap='gray')
    axes[1, 1].set_title('Binary Prediction\n(Threshold = 0.5)')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(input_image, cmap='gray', alpha=0.7)
    axes[1, 2].imshow(binary_pred, cmap='Blues', alpha=0.5)
    axes[1, 2].set_title('Prediction Overlay\n(Blue = Predicted Heart)')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('demo/demo_visualization.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    return input_image, ground_truth, predicted_mask

def calculate_demo_metrics(ground_truth, prediction, threshold=0.5):
    """Calculate metrics for the demo prediction"""
    # Convert to binary
    gt_binary = (ground_truth > 0.5).astype(np.float32)
    pred_binary = (prediction > threshold).astype(np.float32)
    
    # Calculate metrics
    intersection = np.sum(gt_binary * pred_binary)
    union = np.sum(gt_binary) + np.sum(pred_binary) - intersection
    
    dice = (2.0 * intersection + 1e-6) / (np.sum(gt_binary) + np.sum(pred_binary) + 1e-6)
    iou = (intersection + 1e-6) / (union + 1e-6)
    
    # Pixel-level metrics
    tp = np.sum(gt_binary * pred_binary)
    fp = np.sum((1 - gt_binary) * pred_binary)
    fn = np.sum(gt_binary * (1 - pred_binary))
    tn = np.sum((1 - gt_binary) * (1 - pred_binary))
    
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    accuracy = (tp + tn) / (tp + fp + fn + tn)
    
    metrics = {
        'dice_coefficient': float(dice),
        'iou_score': float(iou),
        'precision': float(precision),
        'recall': float(recall),
        'accuracy': float(accuracy),
        'true_positives': int(tp),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'true_negatives': int(tn)
    }
    
    return metrics

def main():
    """Create all demo files"""
    print("🎭 Creating demo sample data for Heart Segmentation CNN...")
    
    # Create demo directory
    demo_dir = Path('demo')
    demo_dir.mkdir(exist_ok=True)
    
    # Generate demo data
    input_image, ground_truth, predicted_mask = create_demo_visualization()
    
    # Save individual components
    np.save('demo/sample_input.npy', input_image)
    np.save('demo/sample_ground_truth.npy', ground_truth)  
    np.save('demo/sample_prediction.npy', predicted_mask)
    
    # Calculate and save metrics
    metrics = calculate_demo_metrics(ground_truth, predicted_mask)
    with open('demo/demo_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Create README for demo
    demo_readme = """# Demo Sample Data

This directory contains sample input and output data for quick verification of the Heart Segmentation CNN model.

## Files:

- `sample_input.npy` - Synthetic chest X-ray image (128×128, normalized to [0,1])
- `sample_ground_truth.npy` - Binary ground truth heart mask (128×128)  
- `sample_prediction.npy` - Expected model prediction probabilities (128×128, [0,1])
- `demo_visualization.png` - Comprehensive visualization of all components
- `demo_metrics.json` - Performance metrics for the demo prediction

## Expected Performance:

The demo sample should achieve approximately:
- **Dice Coefficient:** {dice:.3f}
- **IoU Score:** {iou:.3f}  
- **Accuracy:** {accuracy:.3f}
- **Precision:** {precision:.3f}
- **Recall:** {recall:.3f}

## Usage:

```python
import numpy as np

# Load demo data
input_img = np.load('demo/sample_input.npy')
ground_truth = np.load('demo/sample_ground_truth.npy')
prediction = np.load('demo/sample_prediction.npy')

# Test your trained model
model_prediction = trained_model.predict(input_img[None, ..., None])
```

## Verification:

Use this data to quickly verify that your trained model is working correctly. The synthetic data represents typical heart segmentation challenges and expected performance levels.
""".format(**metrics)
    
    with open('demo/README.md', 'w') as f:
        f.write(demo_readme)
    
    # Print summary
    print("✅ Demo sample data created successfully!")
    print(f"\nDemo metrics:")
    print(f"  • Dice Coefficient: {metrics['dice_coefficient']:.3f}")
    print(f"  • IoU Score: {metrics['iou_score']:.3f}")
    print(f"  • Accuracy: {metrics['accuracy']:.3f}")
    print(f"  • Precision: {metrics['precision']:.3f}")
    print(f"  • Recall: {metrics['recall']:.3f}")
    
    print(f"\nFiles created in demo/ directory:")
    print(f"  • sample_input.npy - Input image")
    print(f"  • sample_ground_truth.npy - Ground truth mask")
    print(f"  • sample_prediction.npy - Expected prediction")
    print(f"  • demo_visualization.png - Visual comparison")
    print(f"  • demo_metrics.json - Performance metrics")  
    print(f"  • README.md - Documentation")

if __name__ == "__main__":
    # Install scipy if not available
    try:
        from scipy import ndimage
    except ImportError:
        print("Installing scipy for demo generation...")
        import subprocess
        subprocess.check_call(['pip', 'install', 'scipy'])
        from scipy import ndimage
    
    main()
