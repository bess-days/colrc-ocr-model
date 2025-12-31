from transformers import TrOCRProcessor, VisionEncoderDecoderModel, Seq2SeqTrainer, Seq2SeqTrainingArguments, default_data_collator
from PIL import Image

model_dir = "./cda-train/checkpoint-15000"
#model_dir = "./cda-train/last_model"
model = VisionEncoderDecoderModel.from_pretrained(model_dir)
processor = TrOCRProcessor.from_pretrained("./cda-train/last_processor")
img = Image.open("./ground-truths/real_03.png").convert("RGB")
#img = Image.open("./to_test/images/testing_03.png").convert("RGB")

pixel_values = processor(images=img, return_tensors="pt").pixel_values
generated_ids = model.generate(pixel_values)
pred_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(pred_text)