import markovify
import random
import os
import re
from augraphy import SubtleNoise, LowInkRandomLines, InkBleed, AugraphyPipeline, BadPhotoCopy, Letterpress
import cv2
from PIL import Image, ImageDraw, ImageFont
import json
from datasets import Dataset
import numpy as np
import unicodedata
random.seed(42)
sp = "ᵃ̈ᶥᵘⁱᵃᵓιᴇɩ"
large_str = ""
characters = set()
for filename in os.listdir("/Users/sbg/colrc-ocr-model/ground-truths"):
        if filename.endswith(".txt"):
            filepath = os.path.join("/Users/sbg/colrc-ocr-model/ground-truths", filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                t = f.read()
                # Normalize the text to NFC form to ensure consistent representation of characters
                t = unicodedata.normalize("NFC", t)
                for char in list(t):
                    # Add the character to the set 
                    characters.add(char)
                full_text = " " + t
                large_str += full_text
words_str = re.sub(r"\.", "", large_str) #remove periods from the string
word_list = set([word.strip() for word in words_str.split(' ') if word.strip()]) #split string into words and remove empty strings
corpus = "\n".join(word_list)
class CharText(markovify.NewlineText):
    # Override the default word_split and word_join methods to treat each character as a "word"
    def word_split(self, sentence):
        return list(sentence)
    def word_join(self, words):
        return "".join(words)

gen_words = []
text_model = CharText(corpus, state_size=3) 
for i in range(1000):
    new_word = text_model.make_sentence(
        tries=50,
        max_words=10,
        min_words=1
    )
    if new_word:
        gen_words.append(new_word)
print("Number of generated words:", len(gen_words))
print("Number of unique generated words:", len(set(gen_words)))
print("Unique generated words:", set(gen_words))
# Add the generated words to the original word list and remove duplicates
gen_words = list(set(gen_words)) + list(word_list)
print("Total number of words after adding generated words:", len(gen_words))
def generate_sentence(words, min_words=6, max_words=8):
    """Generates novel sentences

    Args:
        words (list): the list of words to sample from
        min_words (int, optional): Minimum number of words in the generated sentence. Defaults to 6.
        max_words (int, optional): Maximum number of words in the generated sentence. Defaults to 8.

    Returns:
        str: The generated sentence.
    """
    n_words = random.randint(min_words, max_words)
    punc = random.randint(1, n_words-1)
    sentence = []
    for _ in range(n_words):
        sentence.append(random.choice(words))
    sentence[punc] += "."
    return " ".join(sentence)
# Fonts similar to source text
font_cfg = json.load(open("/Users/sbg/colrc-ocr-model/sources/fonts_config.json"))
all_fonts = font_cfg["Doulos"]
def random_font():
    """Pick random font

    Returns:
        ImageFont: A random font object from the list of available fonts.
    """
    fpath = random.choice(all_fonts)
    size = random.randint(30, 50)
    return ImageFont.truetype(fpath, size=size)

def generate_image(text, font):
    """Generate the base image with the centence is centered, similar font

    Args:
        text (str): The text to be displayed on the image.
        font (ImageFont): The font to be used for the text.

    Returns:
        Image: The generated image.
    """
    temp_img = Image.new("RGB", (1,1))
    draw = ImageDraw.Draw(temp_img)
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    text_w = right - left
    text_h = bottom - top
    img_w = text_w + 20 * 2 
    img_h = text_h + 20 *2
    img = Image.new("RGB", (img_w, img_h), color="white")
    draw = ImageDraw.Draw(img)
    draw.text((20-left, 20-top), text, font=font, fill=0)
    return img
# Define the augmentation pipeline
ink_phase = [
        LowInkRandomLines(count_range=(3, 5),
                                                    use_consistent_lines=True,
                                                    noise_probability=0.1,
                                                    ),
                                            
        InkBleed(intensity_range=(0.5, 0.8),
                    kernel_size=(3, 5),
                    severity=(0.4, 0.6)
                    ),
        Letterpress(n_samples=(100, 200),
                          n_clusters=(100, 300),
                          std_range=(1000, 3000),
                          value_range=(50, 100),
                          value_threshold_range=(50, 50),
                          blur=1
                          ),
]
post_phase = [
    SubtleNoise(subtle_range=25)

    
]
pipeline = AugraphyPipeline(ink_phase=ink_phase, post_phase=post_phase, log=False)


os.makedirs("./gen-samples", exist_ok=True)
for i in range(1000):
    sentence = generate_sentence(gen_words)
    font = random_font()
    image = generate_image(sentence,font)
    path = f"./gen-samples2/sample_{i}.png"
    image.save(path)
    im = cv2.imread(path)
    augmented_image = pipeline(im)
    cv2.imwrite(path, augmented_image)
    with open(f"./gen-samples2/sample_{i}.gt.txt", "w", encoding="utf-8") as f:
        f.write(sentence)
"""
entries = {"image":[], "text":[]}
for i in range(3000):
    sentence = generate_sentence(gen_words)
    font = random_font()
    image = generate_image(sentence,font)
    image = image.convert("RGB")
    aug = np.array(image)
    aug = pipeline(aug)
    aug = Image.fromarray(aug).convert("RGB") 
    entries["text"].append(sentence)
    entries["image"].append(aug)
data = Dataset.from_dict(entries)
data.save_to_disk("./data/")
"""
