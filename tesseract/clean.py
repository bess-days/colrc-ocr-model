import cv2
import numpy as np
from pathlib import Path

# ---------------- CONFIG ----------------
input_image = "./test/images/real_121.png"       # Replace with your image path
output_image = "test_image_clean.png"  # Output path

# ---------------- PREPROCESS FUNCTION ----------------
def preprocess_image(img_path):
    # Load image in grayscale
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not found: {img_path}")

    # 1?? Denoise
    denoised = cv2.fastNlMeansDenoising(img, h=30)

    # 2?? Adaptive Thresholding
    thresh = cv2.adaptiveThreshold(
        denoised, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11, 2
    )

    # 3?? Deskew
    coords = np.column_stack(np.where(thresh > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = thresh.shape
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    deskewed = cv2.warpAffine(thresh, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    # 4?? Remove small specks
    inv = cv2.bitwise_not(deskewed)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
    clean = cv2.morphologyEx(inv, cv2.MORPH_OPEN, kernel)
    clean = cv2.bitwise_not(clean)

    # 5?? Enhance contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    final = clahe.apply(clean)

    return final

# ---------------- RUN PREPROCESS ----------------
clean_image = preprocess_image(input_image)
cv2.imwrite(output_image, clean_image)
print(f"? Processed image saved as {output_image}")
