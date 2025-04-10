# Satellite-Imagery-Based-Cloud-Detection-and-Movement-Prediction
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
