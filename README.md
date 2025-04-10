# Cloud Movement Prediction with Stacked ConvLSTM

This repository contains a detailed end-to-end pipeline for predicting short-term cloud movement using satellite imagery. Our approach seamlessly integrates classical optical flow algorithms with deep learning techniques by incorporating optical flow features into a Stacked ConvLSTM network. The system is designed for robust nowcasting applications in meteorology, renewable energy forecasting, and aviation safety.

## Table of Contents
- [Overview](#overview)
- [Novelty and Contributions](#novelty-and-contributions)
- [Dataset Description](#dataset-description)
- [Preprocessing Pipeline](#preprocessing-pipeline)
- [Optical Flow Algorithms](#optical-flow-algorithms)
- [Model Architecture](#model-architecture)
- [Loss Function](#loss-function)
- [Training Setup](#training-setup)
- [Results](#results)
- [Future Work](#future-work)
- [Usage](#usage)
- [Installation](#installation)
- [Contributing](#contributing)
- [References](#references)
- [License](#license)

## Overview
This project presents a novel pipeline for short-term cloud movement prediction. The system combines classical optical flow methods—such as Lucas-Kanade, Horn-Schunck, Farneback, and DIS Optical Flow—with a deep learning model (Stacked ConvLSTM) using a hybrid loss function composed of Mean Squared Error (MSE) and Structural Similarity Index (SSIM).

Key components include:
- **Optical Flow Calculation:** Computes motion vectors between consecutive satellite images.
- **Spatiotemporal Modeling:** A ConvLSTM network that processes combined image and motion information.
- **Hybrid Loss Function:** Balances pixel-wise accuracy (MSE) with perceptual quality (SSIM).

## Novelty and Contributions
This work introduces a unique approach by combining traditional optical flow methods with modern deep learning architectures. Our contributions include:
- Enriching raw satellite imagery with optical flow data to provide explicit motion cues.
- A Stacked ConvLSTM model tailored for capturing spatiotemporal dependencies in cloud movement.
- A combined loss function that improves both prediction accuracy and perceptual quality.
- Comprehensive evaluation against classical optical flow methods.

## Dataset Description
We utilize satellite images from the GOES-16 ABI with the following characteristics:
- **Source:** GOES-16 ABI L1b Radiances  
- **Spatial Resolution:** 0.5 km (Visible) and 2 km (Infrared)  
- **Temporal Resolution:** Images every 5 minutes  
- **Channels Used:** Visible (Channel 2) and Infrared (Channel 13)  
- **Data Split:** 70% training, 15% validation, 15% testing

## Preprocessing Pipeline
A robust preprocessing pipeline is implemented to ensure high-quality input for training:
- **Radiometric Calibration:** Converts raw digital numbers to brightness temperature or reflectance.
- **Normalization:** Scales pixel values to the [0, 1] range.
- **Cloud Masking:** Generates binary masks to distinguish clouds from the background.
- **Temporal Stacking:** Forms input sequences with six consecutive frames.
- **Optical Flow Computation:** Computes flow vectors between successive frames, appended as extra channels.

## Optical Flow Algorithms
The project evaluates four classical optical flow methods:
- **Lucas-Kanade:** Local, sparse estimation using small windows.
- **Horn-Schunck:** Global, dense estimation with smoothness constraints.
- **Farneback:** Dense estimation via polynomial expansion.
- **DIS Optical Flow (Final Choice):** Provides a balance between accuracy and efficiency.

## Model Architecture
The Stacked ConvLSTM model is designed to capture both spatial and temporal features:
- **Layer Configuration:** Multiple ConvLSTM layers with increasing filter sizes.
- **Input:** Sequences with concatenated image and optical flow channels.
- **Output:** Predicted future cloud image frames.

## Loss Function
A hybrid loss function is used to optimize model performance:
- **Mean Squared Error (MSE):** Measures pixel-level differences.
- **Structural Similarity Index (SSIM):** Focuses on perceptual similarity.
- **Combined Loss:** A weighted sum of MSE and SSIM to balance accuracy and quality.

## Training Setup
Training is carried out on high-performance GPUs (e.g., NVIDIA Tesla V100) with the following hyperparameters:
- **Learning Rate:** 1e-4 (using a cosine annealing schedule)
- **Optimizer:** Adam with standard momentum values
- **Batch Size:** 8
- **Epochs:** 50
- **Data Augmentation:** Includes random flips, rotations, and brightness/contrast adjustments.

## Results
### Quantitative Metrics
| Algorithm       | MSE ↓   | SSIM ↑ | Inference Time (ms/frame) |
|-----------------|---------|--------|---------------------------|
| Lucas-Kanade    | 0.0125  | 0.75   | 120                       |
| Horn-Schunck    | 0.0118  | 0.77   | 2500                      |
| Farneback       | 0.0112  | 0.78   | 300                       |
| **DIS (final)** | **0.0105**  | **0.80**   | **80**                        |

### Qualitative Analysis
Predictions demonstrate:
- Sharper cloud boundaries and smoother transitions.
- High structural fidelity, especially with DIS Optical Flow integrated into the model.

## Future Work
Due to limited computational resources, the current model is constrained in sequence length and depth, limiting its ability to predict beyond 15 frames accurately. Future work will focus on:
- Expanding input sequences and model depth with greater computational power.
- Introducing attention mechanisms and regularization techniques.
- Generalizing to additional spectral bands to further enhance prediction performance.
- Deploying and validating the system in real-world meteorological applications.

## Usage
### Running the Training Pipeline
1. **Clone the Repository:**
   ```bash
   git clone https://github.com/Vimal9900/Satellite-Imagery-Based-Cloud-Detection-and-Movement-Prediction.git
   cd cloud-movement-prediction
