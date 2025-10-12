
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import numpy as np
import random
import cv2
def apply_blur(img):
    blur_values = [5, 7, 9, 11, 13, 15, 17]
    blur_amount = random.choice(blur_values)

    # ensure input is numpy
    if isinstance(img, Image.Image):  
        img = np.array(img)

    blurred_image = cv2.GaussianBlur(img, (blur_amount, blur_amount), 0)

    # convert back to PIL for consistency
    return Image.fromarray(blurred_image)


def apply_exposure(img):
    if isinstance(img, Image.Image):
        img = np.array(img)

    possibilities = [0.5, 0.6, 0.7, 0.8, 0.9]
    brightness_factor = random.choice(possibilities)
    contrast_factor = random.choice(possibilities)

    adjusted_image = cv2.convertScaleAbs(img, alpha=brightness_factor, beta=0)
    adjusted_image = np.clip(adjusted_image * contrast_factor, 0, 255).astype(np.uint8)

    return Image.fromarray(adjusted_image)

def apply_grain(image):
    grain_range = [30, 40, 50, 60, 70, 80, 90]
    grain_intensity = random.choice(grain_range)

    # Convert image to numpy array
    img_array = np.array(image)

    # Generate random noise with the same size as the image
    noise = np.random.randint(-grain_intensity, grain_intensity, size=img_array.shape, dtype='int')

    # Add noise to the image
    noisy_image = np.clip(img_array + noise, 0, 255).astype('uint8')

    # Convert back to PIL image
    noisy_pil_image = Image.fromarray(noisy_image)
    
    return noisy_pil_image
def augment_image(img):
    if random.random() < 0.6:
        img = apply_blur(img)
    if random.random() < 0.6:
        img = apply_exposure(img)
        if random.random() < 0.6:
            img = apply_grain(img)
    return img