from jiwer import cer
from pathlib import Path
import unicodedata

gt_dir = Path("./test/gt")
#pred_dir = Path("./test/pred")
pred_dir = Path("./test/pred")
cers = []

for gt_file in gt_dir.glob("*.gt.txt"):
    base = gt_file.stem.replace(".gt", "")
    pred_file = pred_dir / f"{base}.txt"
    if not pred_file.exists():
        continue
    gt = gt_file.read_text(encoding="utf-8").strip()
    gt = unicodedata.normalize("NFC", gt)
    pred = pred_file.read_text(encoding="utf-8").strip()
    pred = unicodedata.normalize("NFC", pred)
    c = cer(gt, pred)
    cers.append(c)

    print(f"{base}: CER = {c:.4f}")

print("\nAverage CER:", sum(cers) / len(cers))
