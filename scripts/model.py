from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset

from app.generate import get_entries
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


data = get_entries()
train_df, test_df = train_test_split(data, test_size=0.2)
train_df.reset_index(drop=True, inplace=True)
test_df.reset_index(drop=True, inplace=True)
class TextImage(Dataset):
    def __init__(self, root_dir, df, processor, max_target_length=128):
        self.root_dir = root_dir
        self.df = df
        self.processor = processor
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # get file name + text 
        file_name = self.df['image'][idx]
        text = self.df['text'][idx]
        # prepare image (i.e. resize + normalize)
        image = Image.open(self.root_dir + file_name).convert("RGB")
        pixel_values = self.processor(image, return_tensors="pt").pixel_values
        # add labels (input_ids) by encoding the text
        labels = self.processor.tokenizer(text, 
                                          padding="max_length", 
                                          max_length=self.max_target_length).input_ids
        # important: make sure that PAD tokens are ignored by the loss function
        labels = [label if label != self.processor.tokenizer.pad_token_id else -100 for label in labels]

        encoding = {"pixel_values": pixel_values.squeeze(), "labels": torch.tensor(labels)}
        return encoding
