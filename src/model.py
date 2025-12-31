# model.py
import timm
import torch.nn as nn
from torchvision import models

def create_model(num_classes):
    model = models.resnet50(weights="IMAGENET1K_V2")
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    return model

    