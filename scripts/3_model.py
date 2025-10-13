from transformers import TrOCRProcessor, VisionEncoderDecoderModel, Seq2SeqTrainer, Seq2SeqTrainingArguments
from datasets import load_from_disk

processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-stage1")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-stage1")

# Fix missing config
model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id
model.config.max_length = 64

dataset = load_from_disk("synthetic/hf_dataset")

def preprocess(examples):
    pixel_values = [processor(images=img, return_tensors="pt").pixel_values[0] for img in examples["image"]]
    labels = processor.tokenizer(examples["text"], padding="max_length", truncation=True, return_tensors="pt").input_ids
    return {"pixel_values": pixel_values, "labels": labels}

dataset = dataset.map(preprocess, batched=True)
dataset = dataset.train_test_split(test_size=0.2)

training_args = Seq2SeqTrainingArguments(
    output_dir="./trocr-salish",
    per_device_train_batch_size=2,
    num_train_epochs=1,
    logging_steps=100,
    save_steps=1000,
    learning_rate=5e-5,
    save_total_limit=2,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
)
trainer.train().save_model("./trocr-salish")
