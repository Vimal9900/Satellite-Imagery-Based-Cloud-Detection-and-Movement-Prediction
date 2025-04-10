#!/usr/bin/env python3
"""
Satellite Imagery-Based Cloud Movement Prediction(SICMP)
========================================
This script implements a Stacked ConvLSTM model for predicting future frames of cloud images.
The model is trained on a dataset of cloud images, where each image is represented as a sequence of frames.
The model uses optical flow to capture motion information between frames, which is concatenated with the image data.
The training process includes logging metrics and visualizations using wandb, and the model is saved after training.
"""

import os
import glob
import datetime
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
import logging
import wandb

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# ----------------- Configuration -----------------
DATA_DIR = "/csehome/m24mac015/CV_project/Dataset"
MODEL_DIR = "/csehome/m24mac015/CV_project/Model"
PLOTS_DIR = "/csehome/m24mac015/CV_project/CV_Plots"
EPOCHS = 20
BATCH_SIZE = 2
LEARNING_RATE = .00005
T_IN = 3
EXT = "jpg"
HIDDEN_DIM = 64
NUM_LAYERS = 2

# -------- Optical Flow (farneback_flow) --------
def compute_farneback_flow(prev_img, next_img):
    prev = (prev_img * 255).astype(np.uint8)
    next = (next_img * 255).astype(np.uint8)
    if prev.ndim == 3 and prev.shape[2] == 3:
        prev_gray = cv2.cvtColor(prev, cv2.COLOR_RGB2GRAY)
    else:
        prev_gray = prev
    if next.ndim == 3 and next.shape[2] == 3:
        next_gray = cv2.cvtColor(next, cv2.COLOR_RGB2GRAY)
    else:
        next_gray = next
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, next_gray,
        None, 0.5, 3, 15, 3, 5, 1.2, 0
    )
    u, v = flow[..., 0], flow[..., 1]
    return u, v

# -------- Dataset --------
def load_images(folder, ext='jpg'):
    """
    Load and sort images from the given folder based on the timestamp in the filename.
    
    Filenames are expected to follow the format '%d.%m.%Y %H-%M'.
    
    Args:
        folder (str): Path to the image folder.
        ext (str): Image file extension.
        
    Returns:
        list: List of images as numpy arrays normalized between 0 and 1.
    """
    files = glob.glob(os.path.join(folder, f'*.{ext}'))
    if not files:
        raise FileNotFoundError(f"No files with extension '{ext}' found in {folder}")

    def extract_date(file_path):
        base = os.path.basename(file_path)
        name, _ = os.path.splitext(base)
        try:
            return datetime.datetime.strptime(name, '%d.%m.%Y %H-%M')
        except ValueError as e:
            raise ValueError(f"Filename {name} does not match format '%d.%m.%Y %H-%M'") from e

    file_dates = [(f, extract_date(f)) for f in files]
    file_dates.sort(key=lambda x: x[1])
    return [np.array(Image.open(f)) / 255.0 for f, _ in file_dates]

class CloudDataset(Dataset):
    """
    PyTorch Dataset for loading cloud images and computing optical flow.
    """
    def __init__(self, image_dir, T_in=3, ext='jpg'):
        self.images = load_images(image_dir, ext=ext)
        self.T_in = T_in
        self.num_samples = len(self.images) - T_in
        if self.num_samples <= 0:
            raise ValueError("Not enough images for the given T_in")
        # Pre-compute optical flows for consecutive frames
        self.flows = [np.stack(compute_farneback_flow(self.images[i], self.images[i + 1]), axis=0)
                      for i in range(len(self.images) - 1)]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        imgs = []
        flows = []

        for i in range(idx, idx + self.T_in):
            img = self.images[i]
            if img.ndim == 2:
                img = np.expand_dims(img, axis=-1)
            imgs.append(img)

        for i in range(idx, idx + self.T_in - 1):
            flows.append(self.flows[i])

        imgs = np.stack(imgs, axis=0)  # [T_in, H, W, C]
        imgs = np.transpose(imgs, (0, 3, 1, 2))  # [T_in, C, H, W]

        flows = np.stack(flows, axis=0)  # [T_in-1, 2, H, W]
        if flows.shape[0] < self.T_in:
            flows = np.concatenate([flows, flows[-1:]], axis=0)

        # Concatenate images and flows as model input along the channel dimension
        X = np.concatenate([imgs, flows], axis=1)  # [T_in, C+2, H, W]
        y = self.images[idx + self.T_in]
        if y.ndim == 2:
            y = np.expand_dims(y, axis=0)
        else:
            y = np.transpose(y, (2, 0, 1))
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# -------- Model --------
class ConvLSTMCell(nn.Module):
    """
    Basic ConvLSTM cell.
    """
    def __init__(self, input_dim, hidden_dim, kernel_size):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.padding = kernel_size // 2
        self.conv = nn.Conv2d(input_dim + hidden_dim, 4 * hidden_dim, kernel_size, padding=self.padding)

    def forward(self, x, h_cur, c_cur):
        combined = torch.cat([x, h_cur], dim=1)
        conv_output = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.chunk(conv_output, 4, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

class StackedConvLSTM(nn.Module):
    """
    Stacked ConvLSTM model for multi-step prediction.
    """
    def __init__(self, input_dim, hidden_dim, kernel_size=3, num_layers=2):
        super().__init__()
        self.num_layers = num_layers
        self.cells = nn.ModuleList()
        for i in range(num_layers):
            cur_input_dim = input_dim if i == 0 else hidden_dim
            self.cells.append(ConvLSTMCell(cur_input_dim, hidden_dim, kernel_size))
        self.output_layer = nn.Conv2d(hidden_dim, 3, kernel_size=1)

    def forward(self, x_seq):
        B, T, C, H, W = x_seq.shape
        hidden_states = []
        cell_states = []
        # Initialize states for each layer
        for i in range(self.num_layers):
            h = torch.zeros(B, self.cells[i].hidden_dim, H, W).to(x_seq.device)
            c = torch.zeros(B, self.cells[i].hidden_dim, H, W).to(x_seq.device)
            hidden_states.append(h)
            cell_states.append(c)
        # Process each timestep
        for t in range(T):
            x = x_seq[:, t]
            for i in range(self.num_layers):
                h, c = self.cells[i](x, hidden_states[i], cell_states[i])
                hidden_states[i] = h
                cell_states[i] = c
                x = h
        out = self.output_layer(hidden_states[-1])
        return out

# -------- Loss --------
def ssim_loss(pred, target):
    """
    Compute the Structural Similarity Index (SSIM) loss between prediction and target.
    """
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    mu_pred = F.avg_pool2d(pred, kernel_size=3, stride=1, padding=1)
    mu_target = F.avg_pool2d(target, kernel_size=3, stride=1, padding=1)
    sigma_pred = F.avg_pool2d(pred * pred, kernel_size=3, stride=1, padding=1) - mu_pred * mu_pred
    sigma_target = F.avg_pool2d(target * target, kernel_size=3, stride=1, padding=1) - mu_target * mu_target
    sigma_pred_target = F.avg_pool2d(pred * target, kernel_size=3, stride=1, padding=1) - mu_pred * mu_target
    ssim_n = (2 * mu_pred * mu_target + C1) * (2 * sigma_pred_target + C2)
    ssim_d = (mu_pred ** 2 + mu_target ** 2 + C1) * (sigma_pred + sigma_target + C2)
    ssim_map = ssim_n / ssim_d
    return torch.clamp((1 - ssim_map) / 2, 0, 1).mean()

def combined_loss(pred, target, alpha=0.6, beta=0.4):
    """
    Compute combined loss as a weighted sum of MSE and SSIM losses.
    """
    mse = F.mse_loss(pred, target)
    ssim = ssim_loss(pred, target)
    return alpha * mse + beta * ssim

# -------- Training --------
def train(model, train_loader, val_loader, epochs=20, lr=1e-3):
    """
    Train the model with learning rate scheduling and early stopping.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Initialize LR scheduler (ReduceLROnPlateau monitors the validation loss)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                           patience=5, factor=0.5, verbose=True)

    # Initialize early stopping parameters
    best_val_loss = float('inf')
    early_stop_patience = 5
    patience_counter = 0

    # Initialize wandb for logging
    wandb.init(project="CV_Project", config={
        "epochs": epochs,
        "learning_rate": lr,
        "batch_size": train_loader.batch_size,
        "model": "StackedConvLSTM",
    })

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        correct_train = 0
        total_train = 0

        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss = combined_loss(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            correct_train += ((pred - y).abs() < 0.1).float().mean().item()
            total_train += 1

        avg_train_loss = total_train_loss / total_train
        avg_train_acc = correct_train / total_train

        model.eval()
        total_val_loss = 0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                pred = model(X)
                loss = combined_loss(pred, y)
                total_val_loss += loss.item()
                correct_val += ((pred - y).abs() < 0.1).float().mean().item()
                total_val += 1

        avg_val_loss = total_val_loss / total_val
        avg_val_acc = correct_val / total_val

        logger.info(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.6f}, Train Acc: {avg_train_acc:.4f} | "
                    f"Val Loss: {avg_val_loss:.6f}, Val Acc: {avg_val_acc:.4f}")

        # Log metrics to wandb
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "train_acc": avg_train_acc,
            "val_loss": avg_val_loss,
            "val_acc": avg_val_acc,
            "lr": optimizer.param_groups[0]['lr']
        })

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_accs.append(avg_train_acc)
        val_accs.append(avg_val_acc)

        # Step scheduler based on validation loss
        scheduler.step(avg_val_loss)

        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                logger.info("Early stopping triggered.")
                break

    # Create plots and save them locally
    os.makedirs(PLOTS_DIR, exist_ok=True)
    
    loss_plot_path = os.path.join(PLOTS_DIR, "loss_plot.png")
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Training vs Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(loss_plot_path)
    plt.close()

    acc_plot_path = os.path.join(PLOTS_DIR, "accuracy_plot.png")
    plt.figure()
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(val_accs, label='Val Accuracy')
    plt.title('Training vs Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(acc_plot_path)
    plt.close()

    # Log plots as images to wandb
    wandb.log({
        "loss_plot": wandb.Image(loss_plot_path),
        "accuracy_plot": wandb.Image(acc_plot_path)
    })

    wandb.finish()

    return model

# -------- Main Function --------
def main():
    """
    Prepares dataset, model, and starts the training process.
    """
    try:
        dataset = CloudDataset(image_dir=DATA_DIR, T_in=T_IN, ext=EXT)
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        return

    total_len = len(dataset)
    train_size = int(0.8 * total_len)

    train_dataset = Subset(dataset, list(range(train_size)))
    val_dataset = Subset(dataset, list(range(train_size, total_len)))

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Determine input channel dimension (image channels + 2 optical flow channels)
    input_dim = next(iter(train_loader))[0].shape[2]
    model = StackedConvLSTM(input_dim=input_dim, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS)

    model = train(model, train_loader, val_loader, epochs=EPOCHS, lr=LEARNING_RATE)

    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, "cloud_predictor.pth")
    torch.save(model.state_dict(), model_path)
    logger.info(f"Training complete. Model saved at {model_path}")

if __name__ == "__main__":
    main()
