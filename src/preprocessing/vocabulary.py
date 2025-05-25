# src/preprocessing/vocabulary.py

"""
Vocabulary management for tokenizing and numericalization of text
"""

import pickle
from collections import Counter
from tqdm import tqdm
import nltk
from nltk.tokenize import word_tokenize
from typing import List, Dict, Tuple

# Check and download NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
    
# Also download punkt_tab if needed (newer NLTK versions)
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    try:
        nltk.download('punkt_tab', quiet=True)
    except:
        pass  # punkt_tab might not be available in older NLTK versions

# Try to import spaCy for better tokenization
try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
    USE_SPACY = True
except:
    USE_SPACY = False
    print("spaCy not available, using NLTK tokenizer")

from ..utils.constants import PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN
from ..utils.constants import PAD_IDX, SOS_IDX, EOS_IDX, UNK_IDX

class Vocabulary:
    """Vocabulary class to handle text tokenization and numericalization"""
    
    def __init__(self, freq_threshold: int = 5):
        """
        Initialize vocabulary with special tokens
        
        Args:
            freq_threshold: Minimum frequency for a word to be included
        """
        # Special tokens
        self.itos = {
            PAD_IDX: PAD_TOKEN,
            SOS_IDX: SOS_TOKEN,
            EOS_IDX: EOS_TOKEN,
            UNK_IDX: UNK_TOKEN
        }
        self.stoi = {
            PAD_TOKEN: PAD_IDX,
            SOS_TOKEN: SOS_IDX,
            EOS_TOKEN: EOS_IDX,
            UNK_TOKEN: UNK_IDX
        }
        
        # Frequency threshold to include a word in vocabulary
        self.freq_threshold = freq_threshold
        
        # Counter for new indices
        self.idx = 4  # Start after special tokens
        
        # Store word frequencies
        self.word_frequencies = {}
    
    def __len__(self):
        """Get vocabulary size"""
        return len(self.itos)
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text using spaCy or NLTK"""
        if USE_SPACY:
            return [token.text.lower() for token in nlp.tokenizer(text)]
        else:
            return word_tokenize(text.lower())
    
    def build_vocabulary(self, captions: List[str]) -> Dict[str, int]:
        """
        Build vocabulary from a list of captions
        
        Args:
            captions: List of caption strings
            
        Returns:
            frequencies: Dictionary of word frequencies
        """
        # Counter for word frequencies
        frequencies = {}
        
        print(f"Building vocabulary from {len(captions)} captions...")
        
        # Process all captions
        for caption in tqdm(captions):
            # Tokenize caption
            for word in self.tokenize(caption):
                # Update frequency counter
                frequencies[word] = frequencies.get(word, 0) + 1
                
                # Add word to vocab if it reaches threshold
                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = self.idx
                    self.itos[self.idx] = word
                    self.idx += 1
        
        print(f"Built vocabulary with {len(self.itos)} tokens")
        print(f"Added {len(self.itos) - 4} words above frequency threshold {self.freq_threshold}")
        
        # Store word frequencies for analysis
        self.word_frequencies = frequencies
        
        return frequencies
    
    def numericalize(self, text: str) -> List[int]:
        """Convert text to sequence of token indices"""
        tokenized = self.tokenize(text)
        return [
            self.stoi.get(token, self.stoi[UNK_TOKEN])
            for token in tokenized
        ]
    
    def decode(self, indices: List[int]) -> str:
        """Convert a sequence of indices back to text"""
        tokens = [self.itos[idx] for idx in indices if idx < len(self.itos)]
        # Filter out special tokens
        tokens = [token for token in tokens if token not in [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN]]
        return " ".join(tokens)
    
    def save(self, path: str):
        """Save vocabulary to a file"""
        with open(path, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, path: str):
        """Load vocabulary from a file"""
        with open(path, 'rb') as f:
            return pickle.load(f)
    
    def get_most_frequent_words(self, n: int = 100) -> List[Tuple[str, int]]:
        """Get the most frequent words in the vocabulary"""
        counter = Counter(self.word_frequencies)
        return counter.most_common(n)

def preprocess_caption(caption: str) -> str:
    """
    Preprocess caption text
    
    Args:
        caption: Caption string
        
    Returns:
        processed_caption: Processed caption string
    """
    # Convert to lowercase
    caption = caption.lower()
    
    # Tokenize using spaCy or NLTK
    if USE_SPACY:
        tokens = [token.text for token in nlp.tokenizer(caption)]
    else:
        tokens = word_tokenize(caption)
    
    # Join tokens back to string
    return " ".join(tokens)

def analyze_vocab_coverage(df, vocab, caption_col: str = 'processed_caption') -> Tuple[float, Dict[str, int]]:
    """
    Analyze what percentage of words in the dataset are covered by the vocabulary
    
    Args:
        df: DataFrame with captions
        vocab: Vocabulary object
        caption_col: Column name for captions
        
    Returns:
        coverage: Percentage of words covered
        unknown_word_instances: Dictionary of unknown words and counts
    """
    total_words = 0
    unknown_words = 0
    unknown_word_instances = {}
    
    for caption in df[caption_col]:
        tokens = vocab.tokenize(caption)
        total_words += len(tokens)
        
        for token in tokens:
            if token not in vocab.stoi:
                unknown_words += 1
                unknown_word_instances[token] = unknown_word_instances.get(token, 0) + 1
    
    coverage = (total_words - unknown_words) / total_words * 100 if total_words > 0 else 0
    
    print(f"Vocabulary coverage: {coverage:.2f}%")
    print(f"Total words: {total_words}")
    print(f"Unknown words: {unknown_words}")
    
    if unknown_words > 0:
        print("\nTop unknown words:")
        for word, count in sorted(unknown_word_instances.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {word}: {count} occurrences")
    
    return coverage, unknown_word_instances