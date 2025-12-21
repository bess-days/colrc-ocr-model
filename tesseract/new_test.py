from pytesseract import image_to_string
import cv2
from PIL import Image
import sys
from PIL import Image
from pytesseract import Output
import pytesseract
from pdf2image import convert_from_path
import os
import cv2
import numpy as np
import tempfile
def ocr_image(pdf_path, output_path):
    with tempfile.TemporaryDirectory() as path:
        images = convert_from_path(pdf_path=pdf_path, output_folder=path)
    extracted_text = ""
    for i, image in enumerate(images):

        text = image_to_string(image, lang='pho+eng')
        extracted_text += f"--- Page {i + 1} ---\n"
        extracted_text += text + "\n"
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(extracted_text)

ocr_image("./to_test/OriginOfIndianTribes_Typed.pdf", "./to_test/outputs/tribes.txt")