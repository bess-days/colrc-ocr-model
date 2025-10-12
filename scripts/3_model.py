from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import os
from PIL import Image
from datasets import Dataset
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, Seq2SeqTrainer, Seq2SeqTrainingArguments
from datasets import load_from_disk

processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-stage1")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-stage1")
dataset = load_from_disk("synthetic/hf_dataset")
from transformers import AutoTokenizer
def preprocess(examples):
    tokenizer = AutoTokenizer.from_pretrained('microsoft/trocr-base-stage1')
    special_chars = ["č", "ʷ", "ə", "̓", "ɫ", "ä", ""]
    tokenizer.append(special_chars)
    pixel_values = [processor(images=img, return_tensors="pt").pixel_values[0] for img in examples["image"]]
    labels = tokenizer(examples["text"], padding="max_length", truncation=True, return_tensors="pt").input_ids
    return {"pixel_values": pixel_values, "labels": labels}

dataset = dataset.map(preprocess, batched=True)

training_args = Seq2SeqTrainingArguments(
    output_dir="./trocr-salish",
    per_device_train_batch_size=4,
    num_train_epochs=5,
    logging_steps=100,
    save_steps=1000,
    evaluation_strategy="no",
    learning_rate=5e-5,
    save_total_limit=2,
)

trainer = Seq2SeqTrainer(model=model, args=training_args, train_dataset=dataset)
print(trainer.train())
