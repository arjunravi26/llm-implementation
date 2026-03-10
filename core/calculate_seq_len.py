from typing import List
import numpy as np
import re


def calculate_seq_len(texts: List[str], p=90):
    text_lengths = []
    for text in texts:
        text = text.lower()
        text = re.sub(r"[^\w\s]", "", text)
        text_lengths.append(len(text.split()))
    return np.percentile(a=text_lengths, q=p).astype(np.int64)
