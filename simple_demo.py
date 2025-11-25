import numpy as np
import json

# Create synthetic demo data
np.random.seed(42)

# Create 128x128 synthetic chest X-ray image
image = np.zeros((128, 128))
image += np.random.normal(0.1, 0.05, (128, 128))

# Create heart-like structures
y, x = np.ogrid[:128, :128]
heart_center_x, heart_center_y = 64, 70
heart1 = ((x - heart_center_x) ** 2 / 20**2 + (y - heart_center_y) ** 2 / 15**2) < 1
heart_center_x2, heart_center_y2 = 45, 65
heart2 = ((x - heart_center_x2) ** 2 / 12**2 + (y - heart_center_y2) ** 2 / 10**2) < 1
heart_mask = heart1 | heart2

# Add heart regions to image
image[heart_mask] = 0.8 + np.random.normal(0, 0.1, np.sum(heart_mask))
image = np.clip(image, 0, 1)

# Create ground truth
ground_truth = heart_mask.astype(np.float32)

# Create realistic prediction (with some noise)
prediction = ground_truth.astype(np.float32)
noise_mask = np.random.random((128, 128)) > 0.98
prediction[noise_mask] = 0.7
prediction = np.clip(prediction + np.random.normal(0, 0.1, (128, 128)) * ground_truth, 0, 1)

# Save files
np.save('demo/sample_input.npy', image)
np.save('demo/sample_ground_truth.npy', ground_truth)
np.save('demo/sample_prediction.npy', prediction)

# Calculate metrics
gt_binary = (ground_truth > 0.5).astype(np.float32)
pred_binary = (prediction > 0.5).astype(np.float32)
intersection = np.sum(gt_binary * pred_binary)
union = np.sum(gt_binary) + np.sum(pred_binary) - intersection
dice = (2.0 * intersection + 1e-6) / (np.sum(gt_binary) + np.sum(pred_binary) + 1e-6)
iou = (intersection + 1e-6) / (union + 1e-6)

metrics = {
    'dice_coefficient': float(dice),
    'iou_score': float(iou),
    'description': 'Demo sample metrics for Heart Segmentation CNN'
}

with open('demo/demo_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

print("Demo files created successfully!")
print(f"Dice: {dice:.3f}, IoU: {iou:.3f}")
