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
import matplotlib.pyplot as plt
import torch.optim as optim
set_seed(42)
dataset = load_from_disk("./data/")
train_test = dataset.train_test_split(test_size=0.2)
train = train_test["train"]
test = train_test["test"]


class Phonetic(Dataset):
    def __init__(self, root_dir, ds, processor, max_target_length=128):
        self.root_dir = root_dir
        self.ds = ds
        self.processor = processor
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        # get file name + text 
        item = self.ds[idx]
        image = item['image']
        text = item['text']
        # prepare image (i.e. resize + normalize)
        pixel_values = self.processor(image, return_tensors="pt").pixel_values
        # add labels (input_ids) by encoding the text
        labels = self.processor.tokenizer(text, padding="max_length", max_length=self.max_target_length).input_ids
        # important: make sure that PAD tokens are ignored by the loss function
        labels = [label if label != self.processor.tokenizer.pad_token_id else -100 for label in labels]

        encoding = {"pixel_values": pixel_values.squeeze(), "labels": torch.tensor(labels)}
        return encoding
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")
special_tokens = ["č", "ɫ", "ʷ", "u̥","ᵘ", "ɔ", "ä", "ĺ", "ˀ", "ý", "ś", "ᴇ", "x̥","ʙ", "ẃ", "q́","ḿ", "ˠ", "t́", "ʀ", "ᵃ̈", "r̥", "ⁱ",":", "(", ")", "ẃ","ć", "ṕ", "ń", "u̥", "ᵃ" ]
processor.tokenizer.add_tokens(special_tokens)
model.decoder.resize_token_embeddings(len(processor.tokenizer))
train_dataset = Phonetic(root_dir='./',
                           ds=train,
                           processor=processor)
eval_dataset = Phonetic(root_dir='./',
                           ds=test,
                           processor=processor)
# set special tokens used for creating the decoder_input_ids from the labels
model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id
# make sure vocab size is set correctly
model.config.vocab_size = model.config.decoder.vocab_size
generation_config = model.generation_config if hasattr(model, 'generation_config') else GenerationConfig()
generation_config.eos_token_id = processor.tokenizer.sep_token_id
generation_config.max_length = 64
generation_config.early_stopping = True
generation_config.length_penalty = 2.0
generation_config.no_repeat_ngram_size = 3
generation_config.num_beams = 4
model.generation_config = generation_config

from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

training_args = Seq2SeqTrainingArguments(
    predict_with_generate=True,
    eval_strategy="steps",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    fp16=True, 
    output_dir="./cda-train2",
    logging_steps=300,
    save_steps=1500,
    eval_steps=750,
    num_train_epochs=25,
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
trainer.train(resume_from_checkpoint=False)
trainer.save_model(os.path.join("./cda-train2", 'last_model'))
processor.save_pretrained(os.path.join("./cda-train2", 'last_processor'))

