from datasets import load_dataset, load_from_disk, Dataset
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, Seq2SeqTrainer, Seq2SeqTrainingArguments, default_data_collator, enable_full_determinism, set_seed
from PIL import Image
import os
import evaluate
import numpy as np
from torch import nn
import torch
import tensorboard
import random
from transformers import GenerationConfig
from PIL import Image
import torch.optim as optim
import pandas as pd
set_seed(42)
new_df = {"text":[], "file_name":[]}
for filename in os.listdir("./ground-truths"):
        if filename.endswith(".png"):
            filepath = os.path.join("./ground-truths", filename)
            new_df["file_name"].append(filepath)
        if filename.endswith(".txt"):
            filepath = os.path.join("./ground-truths", filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                t = f.read()
                new_df["text"].append(t)
print(len(new_df["text"]), len(new_df["file_name"]))
df = pd.DataFrame.from_dict(new_df)

from sklearn.model_selection import train_test_split

train_df, test_df = train_test_split(df, test_size=0.2)
# we reset the indices to start from zero
train_df.reset_index(drop=True, inplace=True)
test_df.reset_index(drop=True, inplace=True)
import torch
from torch.utils.data import Dataset
from PIL import Image

class Finetune(Dataset):
    def __init__(self, root_dir, df, processor, max_target_length=128):
        self.root_dir = root_dir
        self.df = df
        self.processor = processor
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # get file name + text 
        file_name = self.df['file_name'][idx]
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
processor = TrOCRProcessor.from_pretrained("./cda-train/last_processor")
model = VisionEncoderDecoderModel.from_pretrained("./cda-train/checkpoint-15000")
train_dataset = Finetune(root_dir='./',
                           df=train_df,
                           processor=processor)
eval_dataset = Finetune(root_dir='./',
                           df=test_df,
                           processor=processor)

model.config.vocab_size = model.config.decoder.vocab_size
generation_config = model.generation_config if hasattr(model, 'generation_config') else GenerationConfig()
generation_config.eos_token_id = processor.tokenizer.sep_token_id
generation_config.max_length = 64
generation_config.early_stopping = True
generation_config.length_penalty = 2.0
generation_config.no_repeat_ngram_size = 3
generation_config.num_beams = 4
model.generation_config = generation_config
training_args = Seq2SeqTrainingArguments(
    predict_with_generate=True,
    eval_strategy="steps",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    fp16=True, 
    output_dir="./finetune-train",
    logging_steps=10,
    save_steps=60,
    eval_steps=20,
    num_train_epochs=10,
    report_to="tensorboard",
    remove_unused_columns=False,
    metric_for_best_model='eval_cer',
    load_best_model_at_end=True,
    optim='adafactor'
)
import evaluate

cer_metric = evaluate.load("cer")
wer_metric = evaluate.load("wer")

def compute_metrics(pred):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = processor.tokenizer.pad_token_id
    label_str = processor.batch_decode(labels_ids, skip_special_tokens=True)

    cer = cer_metric.compute(predictions=pred_str, references=label_str)
    wer = wer_metric.compute(predictions=pred_str, references=label_str)

    return {"cer": cer, "wer": wer}
from transformers import default_data_collator

# instantiate trainer
trainer = Seq2SeqTrainer(
    model=model,
    tokenizer=processor,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=default_data_collator
)
trainer.train()




            

