import pandas as pd
import re
import random
import os
from PIL import Image, ImageDraw, ImageFont
import cv2
from utils.image_utils import apply_blur, apply_exposure, apply_grain, augment_image
from utils.text_utils import load_wordlist

salish_words, english_words = load_wordlist("sources/roots.csv", "sources/english.txt")
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
    text = make_text_sample(salish_words, english_words)
    img = Image.new("L", (1400, 100), color='white')
    draw = ImageDraw.Draw(img)
    size = random.randint(24, 50)
    fonts = ["Charis-Regular.ttf", "Charis-Bold.ttf", "Charis-Medium.ttf", "Charis-Italic.ttf",
             "NotoSans-Bold.ttf", "NotoSans-ExtraLight.ttf", "NotoSans-Light.ttf",
             "NotoSans-Medium.ttf", "NotoSans-Regular.ttf", "NotoSans-Thin.ttf",
             "NotoSans-SemiBold.ttf", "DoulosSIL-Regular.ttf"]
    font = ImageFont.truetype(random.choice(fonts), size)
    draw.text((20, 60), text, font=font, fill=0)
    img = augment_image(img)
    img.save(f"synthetic/images/line{i}.png")
    with open(f"synthetic/labels/line{i}.txt", "w", encoding="utf-8") as f:
        f.write(text)