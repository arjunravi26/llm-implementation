import torch
from collections import Counter
from typing import List


class BPE:
    def __init__(self):
        pass


if __name__ == "__main__":
    from core.calculate_seq_len import calculate_seq_len
    data = [
        "Hello, How are you?",
        "What are you doing now?",
        "The smell of fresh rain on dry earth is called petrichor.",
        "Could you please pass the salt?",
        "I'm planning to go for a long walk once the sun sets.",
        "The old lighthouse stood as a silent sentinel against the crashing waves.",
        "A flicker of doubt crossed his face, but he quickly masked it with a smile"
    ]
