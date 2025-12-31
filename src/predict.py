import sys
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from pathlib import Path

from health_logic import health_recommendation

# =========================
# PATHS & CONFIG
# =========================
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "best_model.pth"
DATA_DIR = BASE_DIR / "data" / "train"
DEVICE = "cpu"  # SAFE (your GPU is incompatible)

# =========================
# HELPERS
# =========================
def load_class_names():
    return sorted([d.name for d in DATA_DIR.iterdir() if d.is_dir()])


def load_model(num_classes):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    state_dict = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True
    )
    model.load_state_dict(state_dict)

    model.to(DEVICE)
    model.eval()
    return model


def preprocess_image(img_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    image = Image.open(img_path).convert("RGB")
    return transform(image).unsqueeze(0)


# =========================
# MAIN
# =========================
def main():
    if len(sys.argv) < 2:
        print("Usage: python src/predict.py <image_path>")
        return

    image_path = Path(sys.argv[1])
    if not image_path.exists():
        print("❌ Image not found:", image_path)
        return

    try:
        age = int(input("Enter age: "))
        weight = float(input("Enter weight (kg): "))
        height = float(input("Enter height (meters): "))
    except ValueError:
        print("❌ Invalid input. Please enter numbers.")
        return

    class_names = load_class_names()
    model = load_model(len(class_names))
    image = preprocess_image(image_path).to(DEVICE)

    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1)[0]

    top_prob, top_idx = torch.max(probs, dim=0)
    predicted_class = class_names[top_idx.item()]

    # Health logic
    result = health_recommendation(
        predicted_class=predicted_class,
        age=age,
        weight_kg=weight,
        height_m=height
    )

    # =========================
    # OUTPUT
    # =========================
    print("\n🍽️  FOODVISION HEALTH RESULT")
    print("=" * 35)
    print(f"Image          : {image_path}")
    print(f"Food Detected  : {predicted_class.replace('_',' ').title()}")
    print(f"Confidence     : {top_prob.item()*100:.2f}%")
    print(f"BMI            : {result['bmi']} ({result['bmi_category']})")
    print(f"Calories       : {result['calories']} kcal")
    print(f"Verdict        : {result['verdict']}")
    print(f"Advice         : {result['explanation']}")
    print("=" * 35)


if __name__ == "__main__":
    main()
