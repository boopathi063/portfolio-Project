# src/health_logic.py
import json
from pathlib import Path

# =========================
# PATH SETUP
# =========================
BASE_DIR = Path(__file__).resolve().parent.parent
CALORIES_PATH = BASE_DIR / "configs" / "calories.json"

with open(CALORIES_PATH, "r") as f:
    CALORIES = json.load(f)

# =========================
# BMI LOGIC
# =========================
def compute_bmi(weight_kg, height_m):
    try:
        return round(weight_kg / (height_m ** 2), 2)
    except Exception:
        return None


def bmi_category(bmi):
    if bmi is None:
        return "unknown"
    if bmi < 18.5:
        return "underweight"
    elif bmi < 25:
        return "normal"
    elif bmi < 30:
        return "overweight"
    else:
        return "obese"


# =========================
# FOOD QUALITY SCORING
# =========================
def food_quality(calories):
    if calories <= 200:
        return "healthy"
    elif calories <= 260:
        return "moderate"
    elif calories <= 320:
        return "high"
    else:
        return "very_high"


# =========================
# PROFESSIONAL HEALTH ADVICE
# =========================
def health_recommendation(predicted_class, age, weight_kg, height_m):
    food_key = predicted_class.lower().replace(" ", "_")
    calories = CALORIES.get(food_key)

    bmi = compute_bmi(weight_kg, height_m)
    category = bmi_category(bmi)

    if calories is None:
        return {
            "predicted_class": predicted_class,
            "calories": None,
            "bmi": bmi,
            "bmi_category": category,
            "verdict": "ℹ️ Insufficient Data",
            "explanation": (
                "Nutritional information for this food is currently unavailable. "
                "Consider maintaining balanced portion control."
            ),
            "advice": (
                "Focus on variety and balance in your meals. "
                "Include vegetables, lean proteins, and whole grains."
            )
        }

    quality = food_quality(calories)

    # =========================
    # DECISION MATRIX
    # =========================
    if quality == "healthy":
        verdict = "🟢 Healthy Choice"
        explanation = (
            f"This food is relatively low in calories (~{calories} kcal) "
            "and fits well into a balanced diet."
        )
        advice = (
            "Suitable for regular consumption. "
            "Pair it with protein or fiber-rich foods for optimal nutrition."
        )

    elif quality == "moderate":
        verdict = "🟡 Suitable in Moderation"
        explanation = (
            f"This food provides moderate energy (~{calories} kcal) "
            "and is acceptable when consumed occasionally."
        )
        advice = (
            "Enjoy in controlled portions. "
            "Avoid frequent consumption if weight management is a goal."
        )

    elif quality == "high":
        verdict = "🟠 Limit Intake"
        explanation = (
            f"This food is calorie-dense (~{calories} kcal) "
            "and may contribute to weight gain if consumed frequently."
        )
        advice = (
            "Best reserved for occasional meals. "
            "Balance it with physical activity or lighter meals."
        )

    else:
        verdict = "🔴 Avoid Frequent Consumption"
        explanation = (
            f"This is a high-calorie food (~{calories} kcal) "
            "with limited nutritional benefit."
        )
        advice = (
            "Recommended as an occasional treat only. "
            "Frequent intake may negatively impact metabolic health."
        )

    # =========================
    # BMI-AWARE ADJUSTMENT
    # =========================
    if category in ["overweight", "obese"] and quality in ["high", "very_high"]:
        verdict += " (High Risk)"
        advice += (
            " Given your BMI category, reducing intake of calorie-dense foods "
            "can significantly improve long-term health outcomes."
        )

    if category == "underweight" and quality in ["moderate", "high"]:
        advice += (
            " This food may help support healthy weight gain when paired "
            "with nutrient-dense meals."
        )

    return {
        "predicted_class": predicted_class,
        "calories": calories,
        "bmi": bmi,
        "bmi_category": category,
        "verdict": verdict,
        "explanation": explanation,
        "advice": advice
    }
