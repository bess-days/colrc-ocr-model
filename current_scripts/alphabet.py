import unicodedata
from pathlib import Path
def extract_atomic_alphabet(gt_dir="./ground-truths", out_file="my_alphabet.txt"):
    """
    Normalize GT files, replace problematic sequences with atomic characters,
    and extract a unique alphabet for Tesseract training.
    """
    gt_path = Path(gt_dir)
    out_path = Path(out_file)
    REPLACEMENTS = {
    "r̥": "\uE000",
    "u̥": "\uE001",
    "x̥": "\uE002",
    "ᵃ̈": "\uE010"}
    all_chars = set()

    for gt_file in gt_path.glob("*.gt.txt"):
        text = gt_file.read_text(encoding="utf-8")

        # --- Normalize Unicode to NFC ---
        text = unicodedata.normalize("NFC", text)

        # --- Replace sequences with atomic PUA characters ---
        for seq, repl in REPLACEMENTS.items():
            text = text.replace(seq, repl)

        # --- Save normalized & atomicized GT back ---
        gt_file.write_text(text, encoding="utf-8")

        # --- Add characters to alphabet ---
        for ch in text:
            if ch != "\n":
                all_chars.add(ch)

    # --- Sort and write alphabet to file ---
    sorted_chars = sorted(all_chars, key=lambda c: ord(c))
    with out_path.open("w", encoding="utf-8") as f:
        for ch in sorted_chars:
            f.write(ch + "\n")
    print(f"Alphabet extracted: {len(sorted_chars)} characters")
    print(f"Saved to {out_path}")
if __name__ == "__main__":
    extract_atomic_alphabet(out_file="my_alphabet.txt")