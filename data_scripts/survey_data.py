from collections import Counter
import matplotlib.pyplot as plt
import re
import os
import numpy as np
import unicodedata
import re
import numpy as np
large_str = ""
lengths = []
rpath = "./ground-truths"
#rpath="./gen-samples"
for filename in os.listdir(rpath):
        if filename.endswith(".txt"):
            filepath = os.path.join(rpath, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                t = f.read()
                # Normalize the text to NFC form to ensure consistent representation of characters
                t = unicodedata.normalize("NFC", t)
                full_text = " " + t
                large_str += full_text
                lengths.append(len(t))
print(f"Average character length of ground-truths: {np.mean(lengths)}")
import regex
import unicodedata
from collections import Counter

def grapheme_counter(text):
    """Generate and count all graphemes

    Args:
        text (str): The text to count graphemes in.

    Returns:
        Counter: A counter of all graphemes in the text.
    """
    # Remove whitespace and punctuation from the text
    text = re.sub(r"[\s+'.\)\(\?\-]", "", text)
    text = unicodedata.normalize("NFC", text)
    # Use regex to find all grapheme clusters in the text
    graphemes = regex.findall(r"\X", text)
    return Counter(graphemes)
freqs_counter = grapheme_counter(large_str)
print("Number of unique graphemes (excluding punctuation):", len(freqs_counter))
print("Total grapheme count:", sum(freqs_counter.values()))
print("Most common graphemes:", freqs_counter.most_common(20))
print("Least common graphemes:", freqs_counter.most_common()[-10:])
print("Frequency:", freqs_counter)
def get_zipf_data(freq_counter):
    """Get Zipf Score data to test variety of words

    Args:
        freq_counter (Counter): A counter of all graphemes in the text.

    Returns:
        tuple: A tuple containing the Zipf exponent (alpha), ranks, frequencies, and the frequency counter.
    """
    # Sort frequencies in descending order and compute ranks
    freqs = np.array(sorted(freq_counter.values(), reverse=True))
    ranks = np.arange(1, len(freqs) + 1)
    log_ranks = np.log(ranks)
    log_freqs = np.log(freqs)
    slope, intercept = np.polyfit(log_ranks, log_freqs, 1)
    alpha = -slope
    return alpha, ranks, freqs, freq_counter
def plot_zipf(ranks, freqs):
    """Generates a Zipf Plot

    Args:
        ranks (np.ndarray): An array of ranks.
        freqs (np.ndarray): An array of frequencies.
    """
    # Create a log-log plot of the ranks and frequencies
    plt.figure()
    plt.loglog(ranks, freqs, marker='o', linestyle='none')
    plt.xlabel("Rank")
    plt.ylabel("Frequency")
    plt.title("Zipf Plot of Letter Frequencies")
    plt.show()
alpha, ranks, freqs, counts = get_zipf_data(freqs_counter)

print(f"Estimated Zipf exponent (alpha): {alpha:.2f}")

plot_zipf(ranks, freqs)