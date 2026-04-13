import numpy as np
from typing import List, Dict

class SimpleTokenizer:
    """
    A word-level tokenizer with special tokens.
    """
    
    def __init__(self):
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        self.vocab_size = 0
        
        # Special tokens
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"
    
    def build_vocab(self, texts: List[str]) -> None:
        """
        Build vocabulary from a list of texts.
        Add special tokens first, then unique words.
        """
        # YOUR CODE HERE
        text = []
        for t in texts:
            text.extend(t.lower().split())
        unique_char = set(text)
        unique_char = sorted(unique_char)
        self.id_to_word = {i+4: char for i, char in enumerate(unique_char)}
        self.id_to_word.update({0: "<PAD>", 1: "<UNK>", 2: "<BOS>", 3: "<EOS>"})
        self.word_to_id = {char: i+4 for i, char in enumerate(unique_char)}
        self.word_to_id.update({"<PAD>": 0, "<UNK>": 1, "<BOS>": 2, "<EOS>": 3})
        self.vocab_size = len(self.id_to_word)
    
    def encode(self, text: str) -> List[int]:
        """
        Convert text to list of token IDs.
        Use UNK for unknown words.
        """
        # YOUR CODE HERE
        text = text.lower().split()
        output = [self.word_to_id[t] if t in self.word_to_id.keys() else 1 for t in text]
        return output
    
    def decode(self, ids: List[int]) -> str:
        """
        Convert list of token IDs back to text.
        """
        # YOUR CODE HERE
        text = [self.id_to_word[id] if id in self.id_to_word.keys() else "<UNK>" for id in ids]
        output = " ".join(text)
        return output