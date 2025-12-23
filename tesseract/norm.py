import json

with open("./postocr_mapping.json", encoding="utf-8") as f:
    pua_map = json.load(f)

REV_MAP = {v: k for k, v in pua_map.items()}

def restore_text(text):
    for pua, orig in REV_MAP.items():
        text = text.replace(pua, orig)
    return text

import unicodedata
import re

def normalize_for_eval(text):
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"\s+", " ", text.strip())
    return text
gt_raw = open("test/gt/real_106.gt.txt", encoding="utf-8").read()
ocr_raw = restore_text(open("./out.txt", encoding="utf-8").read())

gt_eval = normalize_for_eval(gt_raw)
ocr_eval = normalize_for_eval(ocr_raw)
print(gt_eval)
print(ocr_eval)
