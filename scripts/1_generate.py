import pandas as pd
import re
import random
import os
from PIL import Image, ImageDraw, ImageFont
import cv2
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import image_utils
from utils import text_utils
import numpy as np

salish_words, english_words = text_utils.load_wordlist("sources/roots.csv", "sources/english.txt")
def make_text_sample(salish_words, english_words):
    salish_sample = " ".join(random.sample(salish_words, k=random.randint(3,5)))
    english_sample = " ".join(random.sample(english_words, k=random.randint(3,5)))
    layouts = [
        salish_sample + random.choice(["."]),
        english_sample + random.choice(["."]),
        salish_sample + random.choice([".", "?"])  + "\n" + english_sample + random.choice([".", "?", "!", "…"])
    ]
    return random.choice(layouts)
os.makedirs("synthetic/images", exist_ok=True)
os.makedirs("synthetic/labels", exist_ok=True)
def random_font():
    fonts = ["Charis-Regular.ttf", "Charis-Bold.ttf", "Charis-Medium.ttf", "Charis-Italic.ttf",
             "NotoSans-Bold.ttf", "NotoSans-ExtraLight.ttf", "NotoSans-Light.ttf",
             "NotoSans-Medium.ttf", "NotoSans-Regular.ttf", "NotoSans-Thin.ttf",
             "NotoSans-SemiBold.ttf", "DoulosSIL-Regular.ttf"]
    fpath = random.choice(fonts)
    size = random.randint(24, 64)
    return ImageFont.truetype(fpath, size=size)

def random_layout(img_w, img_h):
    """return margin and line spacing pattern"""
    margin = random.randint(20, 100)
    spacing = random.randint(10, 40)
    return margin, spacing

def render_text_block(text, font, img_w, img_h):
    img = Image.new("RGB", (img_w, img_h), color="white")
    draw = ImageDraw.Draw(img)
    margin, spacing = random_layout(img_w, img_h)
    y = margin
    for line in text.split("\n"):
        draw.text((margin, y), line, font=font, fill="black")
        bbox = draw.textbbox((0, 0), line, font=font)
        line_height = bbox[3] - bbox[1]
        y += line_height + spacing

    return img

def random_background(img):
    arr = np.array(img, dtype=np.uint8)
    if random.random() < 0.5:
        noise = np.random.randint(0, 30, arr.shape, dtype='uint8')
        arr = np.clip(arr + noise, 0, 255)
    return Image.fromarray(arr)


def make_text_sample2():
    mode = random.choice(["salish", "english", "mixed"])
    if mode == "salish":
        para = text_utils.make_paragraph(salish_words, english_words, lines=random.randint(2, 5))
    elif mode == "english":
        para = text_utils.make_paragraph(english_words, english_words, lines=random.randint(2, 5))
    else:
        mixed = []
        for _ in range(random.randint(2, 5)):
            if random.random() < 0.5:
                mixed.append(text_utils.make_paragraph(salish_words, english_words, lines=1))
            else:
                mixed.append(text_utils.make_paragraph(english_words, salish_words, lines=1))
        para = "\n".join(mixed)
    return para
for i in range(1000):
    text = make_text_sample(salish_words, english_words)
    font = random_font()
    img = render_text_block(text, font, img_w=1400, img_h=400)
    img = image_utils.augment_image(img)
    img.save(f"synthetic/images/sample_{i}.png")
    with open(f"synthetic/labels/sample_{i}.txt", "w", encoding="utf-8") as f:
        f.write(text)