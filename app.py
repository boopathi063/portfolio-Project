import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from pathlib import Path

from src.health_logic import health_recommendation

# =========================
# CONFIG
# =========================
DEVICE = "cpu"
MODEL_PATH = "models/best_model.pth"
DATA_DIR = "data/train"

# =========================
# UTILS
# =========================
def feet_to_meters(feet, inches):
    total_inches = feet * 12 + inches
    return total_inches * 0.0254


@st.cache_resource
def load_class_names():
    return sorted([d.name for d in Path(DATA_DIR).iterdir() if d.is_dir()])


@st.cache_resource
def load_model(num_classes):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    state_dict = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True
    )
    model.load_state_dict(state_dict)
    model.eval()
    return model


def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    return transform(image).unsqueeze(0)

# =========================
# STREAMLIT UI
# =========================
st.set_page_config(
    page_title="FoodVision Health AI",
    page_icon="🍽️",
    layout="centered"
)

st.title("🍽️ FoodVision Health AI")
st.caption("AI-powered food recognition with personalized health advice")

st.divider()

uploaded_file = st.file_uploader(
    "📷 Upload a food image",
    type=["jpg", "jpeg", "png"]
)

st.subheader("👤 Personal Details")

col1, col2, col3 = st.columns(3)
with col1:
    age = st.number_input("Age", 1, 120, 23)

with col2:
    weight = st.number_input("Weight (kg)", 30.0, 200.0, 72.0)

with col3:
    feet = st.number_input("Height (feet)", 3, 8, 5)
    inches = st.number_input("Height (inches)", 0, 11, 8)

height_m = feet_to_meters(feet, inches)

st.divider()

# =========================
# PREDICTION
# =========================
if uploaded_file and st.button("🔍 Analyze Food", use_container_width=True):
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    class_names = load_class_names()
    model = load_model(len(class_names))
    image_tensor = preprocess_image(image)

    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.softmax(outputs, dim=1)[0]

    top3 = torch.topk(probs, 3)

    predicted_class = class_names[top3.indices[0]]
    confidence = top3.values[0].item() * 100

    result = health_recommendation(
        predicted_class=predicted_class,
        age=age,
        weight_kg=weight,
        height_m=height_m
    )

    # =========================
    # RESULTS
    # =========================
    st.subheader("📊 Analysis Result")

    st.markdown(f"""
    **🍽️ Food Detected:** {predicted_class.replace('_',' ').title()}  
    **🎯 Confidence:** {confidence:.2f}%  
    **🧮 BMI:** {result['bmi']} ({result['bmi_category'].title()})  
    **🔥 Calories:** {result['calories']} kcal  
    """)

    if "Not" in result["verdict"]:
        st.error(result["verdict"])
    elif "Moderation" in result["verdict"]:
        st.warning(result["verdict"])
    else:
        st.success(result["verdict"])

    st.info(result["explanation"])

    st.subheader("🔍 Top 3 Predictions")
    for i in range(3):
        st.write(
            f"{i+1}. "
            f"{class_names[top3.indices[i]].replace('_',' ').title()} "
            f"— {top3.values[i].item()*100:.2f}%"
        )

st.divider()
st.caption("⚕️ This application is for educational purposes only and does not replace professional medical advice.")
