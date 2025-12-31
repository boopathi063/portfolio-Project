import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from tqdm import tqdm

from data import get_dataloaders
from utils.early_stopping import EarlyStopping
from utils.checkpoint import save_model

# ---------------- CONFIG ----------------
DATA_DIR = "data"
BATCH_SIZE = 32
IMG_SIZE = 224
EPOCHS = 50
MODEL_PATH = "models/best_model.pth"

DEVICE = "cpu"  # FORCE CPU (RTX 50xx safe)
# ---------------------------------------

def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0

    for imgs, labels in tqdm(loader, desc="Training"):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)

def validate(model, loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return 100 * correct / total

def main():
    train_loader, val_loader, class_names = get_dataloaders(
        DATA_DIR, BATCH_SIZE, IMG_SIZE
    )

    print("Using device:", DEVICE)
    print("Num classes:", len(class_names))

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    model.to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    early_stopping = EarlyStopping(patience=5)
    best_acc = 0

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")

        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion
        )

        val_acc = validate(model, val_loader)

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Validation Accuracy: {val_acc:.2f}%")

        improved = early_stopping.step(val_acc)
        if improved:
            best_acc = val_acc
            save_model(model, MODEL_PATH)

        if early_stopping.should_stop:
            print("\n⏹ Early stopping triggered")
            break

    print(f"\n🏆 Best Validation Accuracy: {best_acc:.2f}%")

if __name__ == "__main__":
    main()
