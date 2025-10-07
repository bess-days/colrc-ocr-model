import pandas as pd
import re
import random
import os
from PIL import Image, ImageDraw, ImageFont
import cv2
from app.image_edit import apply_blur, apply_exposure, apply_grain, augment_image
def get_sources():
    df = pd.read_csv("sources/roots.csv")
    salish_words = df['salish'].to_list()
    with open("sources/english.txt", "r", encoding="utf-8") as f:
        english_words = [line.strip() for line in f if line.strip()]
    return salish_words, english_words
def clean_salish(word):
    return re.sub(r"[-./()/+=·\-\]\[]", "", word)
salish_words_clean = [clean_salish(w[1:-1]) for w in get_sources[0]]
def make_text_sample(salish_words, english_words):
    salish_sample = " ".join(random.sample(salish_words, k=random.randint(3,5)))
    english_sample = " ".join(random.sample(english_words, k=random.randint(3,5)))
    layouts = [
        salish_sample,
        english_sample,
        salish_sample + "\n" + english_sample
    ]
    return random.choice(layouts)
os.makedirs("synthetic/images", exist_ok=True)
os.makedirs("synthetic/labels", exist_ok=True)
for i in range(10):
    text = make_text_sample(salish_words_clean, get_sources()[1])
    img = Image.new("L", (1400, 100), color='white')
    draw = ImageDraw.Draw(img)
    size = random.randint(24, 50)
    fonts = ["Charis-Regular.ttf", "Charis-Bold.ttf", "Charis-Medium.ttf", "Charis-Italic.ttf",
             "NotoSans-Bold.ttf", "NotoSans-ExtraLight.ttf", "NotoSans-Light.ttf",
             "NotoSans-Medium.ttf", "NotoSans-Regular.ttf", "NotoSans-Thin.ttf",
             "NotoSans-SemiBold.ttf", "DoulosSIL-Regular.ttf"]
    font = ImageFont.truetype(random.choice(fonts), size)
    draw.text((10, 10), text, font=font, fill=0)
    img = augment_image(img)
    img.save(f"synthetic/images/line{i}.png")
    with open(f"synthetic/labels/line{i}.txt", "w", encoding="utf-8") as f:
        f.write(text)