# DnCNN Implementation for Image Denoising

## Overview
This repository contains an implementation of the Denoising Convolutional Neural Network (DnCNN) for image denoising. The model has been trained on the MNIST dataset and demonstrates significant improvements in image quality by reducing noise.

## Results
The performance of the model is evaluated using Peak Signal-to-Noise Ratio (PSNR) and Structural Similarity Index Measure (SSIM):

| Metric           | Noisy Image | Denoised Image |
|-----------------|------------|---------------|
| **Average PSNR** | 22.97 dB   | 25.07 dB      |
| **Average SSIM** | 0.6626     | 0.7084        |

## Dataset
The model is trained on the MNIST dataset, which consists of grayscale handwritten digit images. Gaussian noise with σ = 25 was added to simulate real-world noise conditions.

## Model Architecture
The DnCNN model follows a deep residual learning framework:

- **First Layer**: Convolution (3×3) + ReLU activation.
- **Middle Layers**: Multiple convolutional layers (3×3) with batch normalization and ReLU activation.
- **Final Layer**: Convolutional layer (3×3) to predict noise.
- **Residual Learning**: The network learns to estimate noise, which is then subtracted from the input image to obtain the denoised output.

## Requirements
To run this implementation, ensure you have the following dependencies installed:

- Python 3.x
- TensorFlow/Keras
- NumPy
- OpenCV
- Matplotlib

## Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/dncnn.git
   cd dncnn
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Train the model:
   ```bash
   python train.py
   ```

4. Test the model:
   ```bash
   python test.py
   ```

## Acknowledgments
This implementation is based on the DnCNN model introduced in the paper:

- **Zhang et al., "Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising," IEEE Transactions on Image Processing, 2017.**


