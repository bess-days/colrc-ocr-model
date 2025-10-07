import re
import pandas as pd
def clean_salish(word: str) -> str:
    """
    Remove phonetic/morpheme boundary markers from Salish words.
    Adjust regex as needed.
    """
    return re.sub(r"[-./?]", "", word)

def load_wordlist(csv_path, txt_path):

    df = pd.read_csv(csv_path)
    salish = [clean_salish(w) for w in df["Salish"].dropna().tolist()]
    english = []
    with open(txt_path, "r", encoding="utf-8") as f:
        english = [l.strip() for l in f if l.strip()]
    return salish, english