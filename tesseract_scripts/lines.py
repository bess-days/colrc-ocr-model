import cv2
import pytesseract
import numpy as np

# Configuration
FILE_PATH = "./sample_page.png"
LANG_A = "pca"  # Coptic
LANG_B = "eng"  # English

# 1. Load the image
img = cv2.imread(FILE_PATH, cv2.IMREAD_GRAYSCALE)
if img is None:
    print("Error: Could not find image at ./sample_page.png")
    exit()

# 2. Pre-process to find text "blocks"
# We create a mask where ink is white and paper is black
_, mask = cv2.threshold(img, 180, 255, cv2.THRESH_BINARY_INV)

# "Smear" text horizontally to bridge gaps between letters and words
# A (100, 2) kernel is very wide but short, perfect for catching full lines
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (100, 2))
dilated = cv2.dilate(mask, kernel, iterations=1)

# 3. Find and sort line coordinates
contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Extract bounding boxes and filter out small noise/dust
line_boxes = []
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    if h > 15:  # Ignore anything shorter than 15 pixels (noise)
        line_boxes.append((x, y, w, h))

# CRITICAL: Sort from top to bottom based on the Y-coordinate
line_boxes.sort(key=lambda b: b[1])

# 4. Iterate with alternating languages
for idx, (x, y, w, h) in enumerate(line_boxes):
    # Crop with a tiny bit of vertical padding (5px)
    pad = 5
    y1, y2 = max(0, y-pad), min(img.shape[0], y+h+pad)
    x1, x2 = max(0, x-pad), min(img.shape[1], x+w+pad)
    
    line_img = img[y1:y2, x1:x2]
    
    # Determine language: Line 0 = pca, Line 1 = eng, etc.
    current_lang = LANG_A if idx % 2 == 0 else LANG_B
    
    # Run OCR
    # psm 7 = Treat image as a single text line
    text = pytesseract.image_to_string(line_img, lang=current_lang, config='--psm 7')
    
    clean_text = text.strip()
    if clean_text:
        print(f"Line {idx+1} [{current_lang}]: {clean_text}")