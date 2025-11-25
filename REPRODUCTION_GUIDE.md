# Heart Segmentation CNN - Reproduction Instructions

## 🚀 Quick Start Guide

### Prerequisites
- Python 3.7 or higher
- Google Colab account (recommended) or local Jupyter environment
- NIfTI heart segmentation dataset
- GPU access (Tesla T4 or better recommended)

### 📦 Installation

1. **Clone the repository:**
```bash
git clone https://github.com/Rahexa/Heart-Segmentation-Custom-CNN.git
cd Heart-Segmentation-Custom-CNN
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **For Google Colab (Recommended):**
   - Upload `Heart_Segmentation_Custom_CNN.ipynb` to Google Colab
   - Mount your Google Drive containing the HEART_DATASET folder
   - Run all cells sequentially

### 📂 Dataset Setup

1. **Download the NIfTI Heart Dataset** and organize as follows:
```
HEART_DATASET/
├── images/
│   ├── training/     # Training volumes (.nii.gz)
│   └── testing/      # Testing volumes (.nii.gz)
└── segmentations/
    ├── training/     # Training masks (.nii.gz)
    └── testing/      # Testing masks (.nii.gz)
```

2. **Update dataset path** in the notebook:
```python
DATA_DIR = Path("/content/drive/MyDrive/HEART_DATASET")  # Adjust path as needed
```

### 🔬 Reproducing All Experiments

#### Experiment 1: Dropout Regularization
```python
# This will train CustomCNN_Dropout and achieve Dice: 0.6807 ± 0.058
model_dropout = train_experiment_1_dropout()
```

#### Experiment 2: Extra Convolution Layer
```python  
# This will train CustomCNN_ExtraConv and achieve Dice: 0.7038 ± 0.051
model_extra = train_experiment_2_extra_conv()
```

#### Experiment 3: Resolution Scaling
```python
# This will test 240×240 input resolution and achieve Dice: 0.6520 ± 0.062
model_highres = train_experiment_3_resolution()
```

#### Experiment 4: Optimizer Comparison
```python
# This will compare Adam vs RMSprop optimizers
results = train_experiment_4_optimizers()
```

#### Experiment 5: Batch Size Analysis  
```python
# This will compare batch sizes 4 vs 8
results = train_experiment_5_batch_size()
```

### 📊 Expected Results

| Model | Dice Coefficient | IoU Score | Training Loss | Validation Loss |
|-------|------------------|-----------|---------------|-----------------|
| Base CustomCNN | 0.0253 ± 0.0133 | 0.0128 ± 0.0068 | 0.4687 | 0.6677 |
| + Dropout | 0.6807 ± 0.058 | 0.5162 ± 0.072 | 0.3120 | 1.9476 |
| + Extra Conv | 0.7038 ± 0.051 | 0.5431 ± 0.069 | 0.3163 | 1.0096 |
| High Resolution | 0.6520 ± 0.062 | 0.5010 ± 0.074 | 0.3948 | 1.1707 |
| Adam (lr=1e-4) | 0.4125 ± 0.089 | 0.3240 ± 0.098 | 0.5234 | 3.0891 |
| RMSprop | 0.5890 ± 0.067 | 0.4560 ± 0.081 | 0.4621 | 1.6062 |
| Batch Size 8 | 0.6234 ± 0.054 | 0.4890 ± 0.069 | 0.3897 | 0.8811 |

### 🎯 Model Weights and Demo

1. **Best model weights** are automatically saved to `models/best_model.h5`
2. **Demo inference** is available in the final notebook cells
3. **Sample predictions** are saved to `results/images/` for quick verification

### 📈 Visualization and Analysis

All 13 visualization images will be automatically generated and saved:
- `data_sample_check.png` - Dataset visualization
- `base_model_loss_curves.png` - Training curves
- `exp1_prediction_histogram.png` - Prediction analysis
- `exp1_training_comparison.png` - Dropout comparison
- `exp2_catastrophic_failure.png` - Overfitting analysis
- `exp3_resolution_comparison.png` - Resolution effects
- `exp4_optimizer_comparison.png` - Optimizer comparison
- `exp5_batch_comparison.png` - Batch size effects
- And 5 additional detailed analysis images

### 💡 Key Findings to Reproduce

1. **Regularization Impact**: Dropout transforms failure (Dice: 0.0253) into success (Dice: 0.6807)
2. **Overfitting Severity**: All models show train/validation gaps of 0.2-1.6
3. **Parameter Sensitivity**: ~4,700 parameters per training sample is excessive without regularization
4. **Resolution Scaling**: Higher resolution requires proportionally more model capacity

### 🔧 Troubleshooting

**Common Issues:**
1. **Out of Memory**: Reduce batch size from 4 to 2, or use 128×128 resolution only
2. **Dataset Path Error**: Ensure HEART_DATASET is mounted and path is correct
3. **Missing Dependencies**: Run `pip install -r requirements.txt` again
4. **Slow Training**: Use GPU acceleration in Colab (Runtime → Change Runtime Type → GPU)

**Performance Optimization:**
- Use mixed precision training for faster GPU utilization
- Enable XLA compilation for improved performance
- Use tf.data.Dataset for efficient data loading

### 📝 Citation

If you use this code in your research, please cite:
```bibtex
@misc{heart-segmentation-2025,
  title={Heart Segmentation from Chest X-Ray Images Using Custom CNN Architectures},
  author={Raihan Sidker and Sobuj Gupta and Soumen Biswas},
  year={2025},
  publisher={GitHub},
  journal={GitHub repository},
  howpublished={\url{https://github.com/Rahexa/Heart-Segmentation-Custom-CNN}}
}
```

### 📞 Support

For questions or issues:
- Open an issue on GitHub
- Check the comprehensive documentation in the notebook
- Review the LaTeX report for detailed methodology

**Estimated Runtime**: 2-3 hours for all experiments on Tesla T4 GPU
