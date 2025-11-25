# Heart Segmentation from Chest X-Ray Images using Custom CNN

[![Python](https://img.shields.io/badge/Python-3.7%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📋 Project Overview

This project implements **custom Convolutional Neural Network (CNN) architectures** for heart segmentation from chest X-ray and MRI images. Developed as part of the Neural Networks and Fuzzy Logic course (Spring 2025), this work focuses on building CNNs from scratch without using pre-trained models.

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
├── 1128,1100,1110.pdf                     # Project documentation
├── README.md                               # This file
└── .gitignore                             # Git ignore rules
```

## 🚀 Getting Started

### Prerequisites

```bash
pip install tensorflow>=2.0
pip install numpy matplotlib
pip install nibabel
pip install google-colab  # If running on Colab
```

### Usage

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Rahexa/Heart-Segmentation-Custom-CNN.git
   cd Heart-Segmentation-Custom-CNN
   ```

2. **Open the notebook**:
   - For Google Colab: Upload `Heart_Segmentation_Custom_CNN.ipynb`
   - For Jupyter: Run `jupyter notebook Heart_Segmentation_Custom_CNN.ipynb`

3. **Mount your dataset** (if using Colab):
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
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
