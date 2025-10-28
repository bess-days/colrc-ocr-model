import re
import pandas as pd
import random
def clean_salish(word: str) -> str:
    """
    Remove phonetic/morpheme boundary markers from Salish words.
    Adjust regex as needed.
    """
    return re.sub(r"[·/\.\\\?\+=\[\]\-\(\)‑‿]", "", word[1:-1])

def load_wordlist(csv_path, txt_path):
    df = pd.read_csv(csv_path, encoding="utf-8")
    salish = [clean_salish(w) for w in df["salish"].dropna().tolist()]
    english = []
    with open(txt_path, "r", encoding="utf-8") as f:
        english = [l.strip() for l in f if l.strip()]
    return salish, english
def make_sentence(words, length_range=(4, 10)):
    n = random.randint(*length_range)
    sent = " ".join(random.choices(words, k=n))
    sent = sent[0].upper() + sent[1:]
    return sent + random.choice([".", "?", "!", "…"])

def make_paragraph(salish, english, lines=3):
    out = []
    for _ in range(lines):
        if random.random() < 0.5:
            out.append(make_sentence(salish))
        else:
            out.append(make_sentence(english))
    return "\n ".join(out)

