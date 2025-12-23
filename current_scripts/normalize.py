import unicodedata
from pathlib import Path
import json

def process_gt(gt_dir="./norm_ground",
               alphabet_file="my_alphabet.txt",
               postocr_mapping_file="postocr_mapping.json"):
    """
    Normalize GT files, apply atomic replacements (PUA), extract alphabet,
    and save a post-OCR mapping.
    """
    gt_path = Path(gt_dir)
    alphabet_path = Path(alphabet_file)
    mapping_path = Path(postocr_mapping_file)

    # --- Define atomic replacements (sequence -> PUA) ---
    ATOMIC_REPLACEMENTS = {
        "r̥": "\uE000",
    "u̥": "\uE001",
    "x̥": "\uE002",
    "ᵃ̈": "\uE010"
    }

    all_chars = set()

    gt_files = list(gt_path.glob("*.gt.txt"))
    if not gt_files:
        print(f"No .gt.txt files found in {gt_dir}")
        return

    for gt_file in gt_files:
        text = gt_file.read_text(encoding="utf-8")

        # --- Normalize Unicode to NFC ---
        text = unicodedata.normalize("NFC", text)

        # --- Apply atomic PUA replacements ---
        for seq, repl in ATOMIC_REPLACEMENTS.items():
            text = text.replace(seq, repl)

        # --- Save normalized & atomicized GT ---
        gt_file.write_text(text, encoding="utf-8")

        # --- Add characters to alphabet ---
        for ch in text:
            if ch != "\n":
                all_chars.add(ch)

        print(f"Processed: {gt_file.name}")

    # --- Save alphabet sorted by Unicode ---
    sorted_chars = sorted(all_chars, key=lambda c: ord(c))
    with alphabet_path.open("w", encoding="utf-8") as f:
        for ch in sorted_chars:
            f.write(ch + "\n")
    print(f"Alphabet saved to {alphabet_path} ({len(sorted_chars)} characters)")

    # --- Save post-OCR mapping for readability ---
    with mapping_path.open("w", encoding="utf-8") as f:
        json.dump(ATOMIC_REPLACEMENTS, f, ensure_ascii=False, indent=2)
    print(f"Post-OCR mapping saved to {mapping_path}")

# --- Example usage ---
if __name__ == "__main__":
    process_gt(
        alphabet_file="./my_alphabet.txt",
        postocr_mapping_file="./postocr_mapping.json"
    )
