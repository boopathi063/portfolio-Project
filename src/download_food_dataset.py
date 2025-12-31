from bing_image_downloader import downloader
import os

FOOD_CLASSES = [
    "pizza", "hamburger", "hot dog", "french fries", "fried rice",
    "spaghetti", "ramen", "sushi", "steak", "grilled chicken",
    "salad", "caesar salad", "sandwich", "tacos",
    "pancakes", "waffles", "ice cream", "donuts",
    "apple pie", "cheesecake"
]

BASE_DIR = "data/train"

os.makedirs(BASE_DIR, exist_ok=True)

for food in FOOD_CLASSES:
    print(f"Downloading: {food}")
    downloader.download(
        food,
        limit=120,          # images per class
        output_dir=BASE_DIR,
        adult_filter_off=True,
        force_replace=False,
        timeout=60
    )

print("✅ Food dataset downloaded successfully!")
