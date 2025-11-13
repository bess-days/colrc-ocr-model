from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch

# Paths
model_dir = "scripts/trocr-cda/checkpoint-7500"
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
model = VisionEncoderDecoderModel.from_pretrained(model_dir)
img = Image.open("./test/line_screenshot.png").convert("RGB")

pixel_values = processor(images=img, return_tensors="pt").pixel_values
generated_ids = model.generate(pixel_values)
pred_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

print("🔹 Predicted:", pred_text)
