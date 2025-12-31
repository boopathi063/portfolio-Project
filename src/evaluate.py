import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from pathlib import Path
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import itertools

# ---------------- CONFIG ----------------
DATA_DIR = "data/val"
MODEL_PATH = "models/best_model.pth"
BATCH_SIZE = 32
DEVICE = "cpu"  # keep CPU safe
IMG_SIZE = 224

# ---------------- UTILS ----------------
def load_class_names():
    return sorted([d.name for d in Path(DATA_DIR).iterdir() if d.is_dir()])

def get_dataloader():
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])

    dataset = ImageFolder(DATA_DIR, transform=transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    return loader, dataset.classes

def load_model(num_classes):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model

# ---------------- EVALUATION ----------------
def evaluate():
    loader, class_names = get_dataloader()
    model = load_model(len(class_names))

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Accuracy
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels)) * 100
    print(f"\n✅ Validation Accuracy: {accuracy:.2f}%")

    # Classification Report
    print("\n📊 Classification Report:\n")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plot_confusion_matrix(cm, class_names)

# ---------------- PLOTTING ----------------
def plot_confusion_matrix(cm, classes):
    plt.figure(figsize=(12, 10))
    plt.imshow(cm, interpolation='nearest')
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    Path("results").mkdir(exist_ok=True)
    plt.savefig("results/confusion_matrix.png")
    print("📁 Confusion matrix saved to results/confusion_matrix.png")
    plt.show()


if __name__ == "__main__":
    evaluate()
