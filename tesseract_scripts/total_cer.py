from jiwer import cer
from pathlib import Path
import unicodedata

gt_dir = Path("./test/gt")
pred_dir = Path("./test/pred-base")
#pred_dir = Path("./test/pred")
cers = []
pred_str = ""
gt_str = ""
for gt_file in gt_dir.glob("*.gt.txt"):
    base = gt_file.stem.replace(".gt", "")
    pred_file = pred_dir / f"{base}.txt"
    if not pred_file.exists():
        continue
    gt = gt_file.read_text(encoding="utf-8").strip()
    gt = unicodedata.normalize("NFC", gt)
    pred = pred_file.read_text(encoding="utf-8").strip()
    pred = unicodedata.normalize("NFC", pred)
    pred_str += f" {pred}"
    gt_str += f" {gt}"
print(cer(gt_str, pred_str))