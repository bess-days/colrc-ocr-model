from PIL import Image
from pytesseract import Output
import pytesseract
from pdf2image import convert_from_path
import os
from pdf2image import pdfinfo_from_path
import cv2
import numpy as np
import tempfile
#print(pytesseract.image_to_string("test/pages/CDHOC_Typed_Images2.png", lang="cda+eng"))
print(pytesseract.image_to_string("test/pages/CoyoteSnaresTheWind_Typed_Images2.png", lang="cda+eng"))