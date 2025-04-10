##1 Satellite-Imagery-Based-Cloud-Detection-and-Movement-Prediction
This repository contains a detailed, end-to-end pipeline for detecting clouds and predicting their movement using satellite imagery.

Our approach integrates classi-
cal computer vision algorithms with modern deep learning techniques. We evaluate
and compare four prominent optical flow methods—Lucas-Kanade, Horn-Schunck,
Farneback, and DIS Optical Flow. Among these, DIS was selected for its balance
of accuracy and computational efficiency. The optical flow maps are then used
in conjunction with a stacked ConvLSTM network trained using a combined loss
function of Mean Squared Error (MSE) and Structural Similarity Index (SSIM).
Results show that our method achieves accurate short-term cloud movement pre-
dictions with high temporal coherence.

##2 Dataset Description
We use satellite images from the GOES-16 satellite, particularly its Advanced Baseline
Imager (ABI). The dataset characteristics are as follows:
• Source: GOES-16 ABI L1b Radiances.
• Spatial Resolution: 0.5 km (VIS) and 2 km (IR).
• Temporal Resolution: Every 5 minutes.
• Channels Used: Visible (Ch 2) and Infrared (Ch 13).
• Ground Truth: Derived using thresholding (e.g., brightness temperature ¡ 230K).
• Data Split: 70% training, 15% validation, 15% testing.
Images are collected from daylight and nighttime sequences across seasons to ensure
generalization. All data is preprocessed to correct for radiometric calibration and stan-
dardized spatial resolution.
