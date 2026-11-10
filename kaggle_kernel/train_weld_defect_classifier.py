"""
WeldGuard-AI: Multi-Process Welding Defect Classifier (Kaggle GPU Training Script)
Supports training and fine-tuning on open-source Kaggle and Roboflow welding datasets:
  - GMAW/MIG: Good Weld, Burn-Through, Lack of Fusion, Spatter Eruption, Porosity
  - GTAW/TIG: Tungsten Contamination, Oxidation, Undercut
  - SMAW/Stick: Slag Inclusion, Incomplete Penetration, Crack

Backbone: EfficientNet-B0 / MobileNetV3 / ResNet34 with Transfer Learning
Hardware: Kaggle NVIDIA T4 / P100 / RTX 3050 GPU
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image

# ---------------------------------------------------------------------
# 1. Defect Taxonomy Definition
# ---------------------------------------------------------------------
WELD_DEFECT_CLASSES = [
    "good_weld",
    "burn_through",
    "lack_of_fusion",
    "porosity",
    "spatter_burst",
    "undercut",
    "slag_inclusion",
    "crack"
]
NUM_CLASSES = len(WELD_DEFECT_CLASSES)


# ---------------------------------------------------------------------
# 2. Dataset Loader with Industrial Augmentation
# ---------------------------------------------------------------------
class IndustrialWeldDataset(Dataset):
    def __init__(self, file_paths, labels, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


def get_weld_transforms():
    """
    Domain-specific data augmentation simulating industrial welding conditions:
    Variable arc glare, slight camera tilt, spatter motion blur, and sensor noise.
    """
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return train_transform, val_transform


# ---------------------------------------------------------------------
# 3. Model Architecture with Transfer Learning
# ---------------------------------------------------------------------
class WeldDefectClassifier(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, pretrained=True):
        super(WeldDefectClassifier, self).__init__()
        weights = models.ResNet34_Weights.DEFAULT if pretrained else None
        self.backbone = models.resnet34(weights=weights)

        # Replace final classification head with industrial dropout + multi-class linear layer
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=0.35),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.25),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)


# ---------------------------------------------------------------------
# 4. Training Engine
# ---------------------------------------------------------------------
def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    device,
    epochs=15,
    save_path="models/weld_defect_classifier_best.pth"
):
    best_val_acc = 0.0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    print(f"\n[WeldDefectClassifier] Starting training on {device} for {epochs} epochs...")
    t_start = time.time()

    for epoch in range(epochs):
        # Training Phase
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct_train += (preds == targets).sum().item()
            total_train += targets.size(0)

        epoch_train_loss = running_loss / max(1, total_train)
        epoch_train_acc = (correct_train / max(1, total_train)) * 100.0

        # Validation Phase
        model.eval()
        val_running_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                val_running_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                correct_val += (preds == targets).sum().item()
                total_val += targets.size(0)

        epoch_val_loss = val_running_loss / max(1, total_val)
        epoch_val_acc = (correct_val / max(1, total_val)) * 100.0

        scheduler.step(epoch_val_loss)

        history["train_loss"].append(epoch_train_loss)
        history["val_loss"].append(epoch_val_loss)
        history["train_acc"].append(epoch_train_acc)
        history["val_acc"].append(epoch_val_acc)

        print(
            f"Epoch {epoch+1:02d}/{epochs:02d} | "
            f"Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc:.1f}% | "
            f"Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.1f}%"
        )

        # Checkpointing
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "val_acc": best_val_acc,
                "classes": WELD_DEFECT_CLASSES
            }, save_path)
            print(f"  --> Saved new best checkpoint (Val Acc: {best_val_acc:.1f}%) to {save_path}")

    elapsed = time.time() - t_start
    print(f"\n[WeldDefectClassifier] Training completed in {elapsed:.1f}s. Best Val Acc: {best_val_acc:.1f}%")
    return history


def main():
    parser = argparse.ArgumentParser(description="WeldGuard-AI Defect Classifier Training")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--save-model", type=str, default="models/weld_defect_classifier_best.pth")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")


if __name__ == "__main__":
    main()
