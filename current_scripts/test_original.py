from transformers import TrOCRProcessor, VisionEncoderDecoderModel, Seq2SeqTrainer, Seq2SeqTrainingArguments, default_data_collator
from PIL import Image

model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
img = Image.open("./ground-truths/real_01.png").convert("RGB")
pixel_values = processor(images=img, return_tensors="pt").pixel_values
generated_ids = model.generate(pixel_values)
pred_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(pred_text)