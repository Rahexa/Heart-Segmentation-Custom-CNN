# Heart Segmentation from Chest X-Ray Images using Custom CNN

[![Python](https://img.shields.io/badge/Python-3.7%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📋 Project Overview

This project implements **custom Convolutional Neural Network (CNN) architectures** for heart segmentation from chest X-ray and MRI images. Developed as part of the Neural Networks and Fuzzy Logic course (Spring 2025), this work focuses on building CNNs from scratch without using pre-trained models.

## 🏆 Key Research Contributions

- **Regularization Impact**: Demonstrated that dropout regularization can transform complete segmentation failure (Dice: 0.0253) into functional performance (Dice: 0.6807) - a 2590% improvement
- **Parameter-Data Ratio Analysis**: Established critical thresholds for model capacity in data-limited medical imaging scenarios (~4,700 parameters per training sample proved excessive)
- **Systematic Overfitting Study**: Comprehensive analysis of overfitting patterns across 7 different architectural variants
- **Educational Deep Learning**: Complete implementation from fundamental principles without transfer learning

## 🎯 Objectives

- **Custom Architecture Development**: Build fully custom CNN models without relying on pre-built architectures (ResNet, VGG, etc.)
- **Comprehensive Experimentation**: Conduct in-depth analysis of architecture variations, hyperparameter tuning, and input size modifications
- **Qualitative Analysis**: Visualize feature maps, prediction masks, and training behavior
- **Medical Image Processing**: Process NIfTI format medical images for 2D segmentation tasks

## 🏗️ Architecture Features

### Custom CNN Components
- **Convolutional Layers**: Custom filter designs and configurations
- **Pooling Strategies**: Various pooling techniques for feature extraction
- **Activation Functions**: Optimized activation choices for medical imaging
- **Skip Connections**: Custom implementation for better gradient flow
- **Attention Mechanisms**: Custom attention modules for improved segmentation

### Experimental Variations
- Multiple architecture configurations
- Different input resolutions and preprocessing techniques
- Various loss functions optimized for segmentation
- Comprehensive hyperparameter exploration

## 📊 Dataset

**Source**: NIfTI format medical imaging dataset

**Structure**:
```
HEART_DATASET/
├── images/
│   ├── training/     # Training volumes (.nii.gz)
│   └── testing/      # Testing volumes (.nii.gz)
└── segmentations/
    ├── training/     # Training masks (.nii.gz)
    └── testing/      # Testing masks (.nii.gz)
```

**Processing**: 
- Extracts middle 2D axial slices from 3D volumes
- Converts to 2D segmentation format
- Multiple annotation support (up to 3 annotations per image)

## 🛠️ Technologies Used

- **Python 3.7+**
- **TensorFlow 2.x** - Deep learning framework
- **NumPy** - Numerical computations
- **Matplotlib** - Visualization and plotting
- **NiBabel** - NIfTI file processing
- **Google Colab** - Development environment

## 📁 Project Structure

```
Heart-Segmentation-Custom-CNN/
├── Heart_Segmentation_Custom_CNN.ipynb    # Main notebook with all implementations
├── train.py                               # Standalone training script
├── inference.py                           # Model inference script
├── requirements.txt                       # Python dependencies
├── REPRODUCTION_GUIDE.md                  # Detailed reproduction instructions
├── demo/                                  # Demo files for quick verification
│   ├── sample_input.npy                   # Sample input image
│   ├── sample_ground_truth.npy            # Sample ground truth mask
│   ├── sample_prediction.npy              # Expected model output
│   ├── demo_metrics.json                  # Performance metrics
│   └── README.md                          # Demo documentation
├── models/                                # Trained model weights (created during training)
├── results/                               # Training results and visualizations
├── HEART_DATASET/                         # Dataset directory (user provided)
├── Heart_Segmentation_Complete_Report.tex # Complete LaTeX research report
├── 1128,1100,1110.pdf                    # Original project documentation
└── README.md                              # This file
```

## 🚀 Getting Started

### Quick Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Rahexa/Heart-Segmentation-Custom-CNN.git
   cd Heart-Segmentation-Custom-CNN
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Quick demo verification**:
   ```bash
   python -c "import numpy as np; print('Demo input shape:', np.load('demo/sample_input.npy').shape)"
   ```

### Training Options

#### Option 1: Jupyter Notebook (Recommended)
- Upload `Heart_Segmentation_Custom_CNN.ipynb` to Google Colab
- Follow the comprehensive experimental pipeline
- All visualizations and analyses included

#### Option 2: Standalone Training Script
```bash
# Train specific experiment
python train.py --experiment dropout --epochs 20

# Train all experiments
python train.py --experiment all

# Custom configuration
python train.py --experiment extra_conv --batch_size 4 --learning_rate 1e-3
```

#### Option 3: Model Inference Only
```bash
# Run inference on new images
python inference.py --input_path path/to/image.nii.gz --model_path models/best_model.h5
```

4. **Update dataset path** in the notebook to your dataset location

## 🔬 Experiments Conducted

### Architecture Variations
- **Baseline CNN**: Simple convolutional architecture
- **Deep CNN**: Increased depth with regularization
- **U-Net Style**: Encoder-decoder with skip connections
- **Attention-Enhanced**: Custom attention mechanisms
- **Multi-Scale**: Multiple input resolution processing

### Hyperparameter Studies
- Learning rate optimization
- Batch size analysis
- Optimizer comparisons (Adam, SGD, RMSprop)
- Loss function evaluation (Dice, IoU, Binary Cross-entropy)

### Analysis Components
- **Feature Map Visualization**: Understanding learned representations
- **Training Dynamics**: Loss curves and convergence analysis
- **Prediction Quality**: Qualitative mask assessment
- **Performance Metrics**: Quantitative evaluation (Dice coefficient, IoU, accuracy)

## 📈 Results

The notebook contains comprehensive results including:
- Model performance comparisons
- Training convergence analysis
- Qualitative segmentation results
- Feature map visualizations
- Architecture ablation studies

*Detailed results and analysis are available in the main notebook.*

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

**Team Members**: [Add your team member names and IDs here]

## 🙏 Acknowledgments

- Neural Networks and Fuzzy Logic Course, Spring 2025
- Dataset providers for medical imaging data
- TensorFlow and open-source community

## 📧 Contact

For questions or collaboration opportunities, please reach out through:
- GitHub Issues: [Create an issue](https://github.com/Rahexa/Heart-Segmentation-Custom-CNN/issues)
- Repository: [Heart-Segmentation-Custom-CNN](https://github.com/Rahexa/Heart-Segmentation-Custom-CNN)

---

**Note**: This project is for educational purposes as part of academic coursework. Please ensure proper attribution when using or referencing this work.
