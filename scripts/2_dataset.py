from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import os
from PIL import Image
from datasets import Dataset
def load_examples():
    for img_file in os.listdir("synthetic/images"):
        if not img_file.endswith(".png"): continue
        text_file = "synthetic/labels/" + img_file.replace(".png", ".txt")
        yield {
            "image": Image.open("synthetic/images/" + img_file).convert("RGB"),
            "text": open(text_file, encoding="utf-8").read().strip()
        }

dataset = Dataset.from_generator(load_examples)
dataset.save_to_disk("synthetic/hf_dataset")
print("? Dataset saved to synthetic/hf_dataset")