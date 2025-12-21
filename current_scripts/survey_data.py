from collections import Counter
import matplotlib.pyplot as plt
import re
import os
import numpy as np
import unicodedata
large_str = ""
for filename in os.listdir("./ground-truths"):
        if filename.endswith(".txt"):
            filepath = os.path.join("./ground-truths", filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                t = f.read()
                full_text = " " + t
                large_str += full_text
text = large_str
words_str = re.sub(r"\.", "", large_str)
word_list = set([word.strip() for word in words_str.split(' ') if word.strip()])
corpus = "\n".join(word_list)
corpus = unicodedata.normalize("NFC", text)
letters = [ch for ch in corpus if ch.isalpha()]
print(len(letters))
freqs = Counter(letters)
print(freqs)
sorted_freqs = sorted(freqs.values(), reverse=True)
sorted_items = freqs.most_common()
letters_sorted = [l for l, f in sorted_items]
counts_sorted = [f for l, f in sorted_items]
ranks = range(1, len(counts_sorted) + 1)
top_n = 20

plt.figure()
plt.loglog(ranks, counts_sorted, marker='o')

for rank, freq, letter in zip(ranks[:top_n], counts_sorted[:top_n], letters_sorted[:top_n]):
    plt.annotate(
        letter,
        (rank, freq),
        xytext=(5, 5),
        textcoords="offset points",
        fontsize=10,
        fontweight="bold"
    )

plt.xlabel("Rank")
plt.ylabel("Frequency")
plt.title("Zipf Curve (Top Letters Labeled)")
plt.show()
plt.figure()
plt.loglog(ranks, counts_sorted, marker='o')

# Label each point
for rank, freq, letter in zip(ranks, counts_sorted, letters_sorted):
    plt.text(rank, freq, letter, fontsize=9)

plt.xlabel("Rank")
plt.ylabel("Frequency")
plt.title("Zipf Curve with Letter Labels")
plt.show()
ranks = range(1, len(sorted_freqs) + 1)
log_ranks = np.log(ranks)
log_freqs = np.log(sorted_freqs)
slope, intercept = np.polyfit(log_ranks, log_freqs, 1)
print(f"Estimated Zipf exponent: {slope:.2f}")



