from transformers import TrOCRProcessor, VisionEncoderDecoderModel, Seq2SeqTrainer, Seq2SeqTrainingArguments, default_data_collator
from PIL import Image

model_dir = "./finetune-train/checkpoint-60"
model = VisionEncoderDecoderModel.from_pretrained(model_dir)
processor = TrOCRProcessor.from_pretrained("./cda-train/last_processor")
img = Image.open("./to_test/images/testing_03.png").convert("RGB")

pixel_values = processor(images=img, return_tensors="pt").pixel_values
generated_ids = model.generate(pixel_values)
pred_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(pred_text)