import os
import json
import pickle
import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import List, Dict, Set, Optional, Tuple, Counter as CounterType
from collections import Counter
from pathlib import Path
from datetime import datetime
from datasets import load_dataset
from tqdm.auto import tqdm
import time
import random
from sklearn.manifold import TSNE

class TurkishMorphology:
    """Turkish morphological analyzer for better tokenization"""
    
    def __init__(self):
        self.vowels = set('aeıioöuüâîû')
        self.consonants = set('bcçdfgğhjklmnprsştvyz')
        
        # Vowel harmony rules
        self.vowel_harmony_map = {
            'a': ['ı', 'a'], 'e': ['i', 'e'],
            'ı': ['ı', 'a'], 'i': ['i', 'e'],
            'o': ['u', 'a'], 'ö': ['ü', 'e'],
            'u': ['u', 'a'], 'ü': ['ü', 'e'],
            'â': ['ı', 'a'], 'î': ['i', 'e'], 'û': ['u', 'a']
        }
        
        # Common Turkish suffixes by category
        self.suffixes = {
            'case': {
                'accusative': ['ı', 'i', 'u', 'ü', 'yı', 'yi', 'yu', 'yü'],
                'dative': ['a', 'e', 'ya', 'ye'],
                'locative': ['da', 'de', 'ta', 'te'],
                'ablative': ['dan', 'den', 'tan', 'ten'],
                'genitive': ['ın', 'in', 'un', 'ün', 'nın', 'nin', 'nun', 'nün']
            },
            'plural': ['lar', 'ler'],
            'possession': {
                'p1s': ['m', 'ım', 'im', 'um', 'üm'],
                'p2s': ['n', 'ın', 'in', 'un', 'ün'],
                'p3s': ['ı', 'i', 'u', 'ü', 'sı', 'si', 'su', 'sü'],
                'p1p': ['mız', 'miz', 'muz', 'müz', 'ımız', 'imiz', 'umuz', 'ümüz'],
                'p2p': ['nız', 'niz', 'nuz', 'nüz', 'ınız', 'iniz', 'unuz', 'ünüz'],
                'p3p': ['ları', 'leri']
            },
            'tense': {
                'present': ['yor', 'iyor', 'ıyor', 'uyor', 'üyor'],
                'past': ['dı', 'di', 'du', 'dü', 'tı', 'ti', 'tu', 'tü'],
                'future': ['acak', 'ecek', 'yacak', 'yecek', 'acağ', 'eceğ'],
                'aorist': ['ar', 'er', 'ır', 'ir', 'ur', 'ür', 'z'],
                'conditional': ['sa', 'se'],
                'necessity': ['malı', 'meli'],
                'ability': ['abil', 'ebil', 'yabil', 'yebil']
            },
            'derivation': {
                'infinitive': ['mak', 'mek'],
                'participle': ['an', 'en', 'yan', 'yen', 'dık', 'dik', 'duk', 'dük'],
                'gerund': ['arak', 'erek', 'ip', 'ıp', 'up', 'üp', 'ince', 'ınca'],
                'passive': ['ıl', 'il', 'ul', 'ül', 'n'],
                'causative': ['tır', 'tir', 'tur', 'tür', 'dır', 'dir', 'dur', 'dür'],
                'negative': ['ma', 'me']
            },
            'common': ['de', 'da', 'ki', 'mi', 'mı', 'mu', 'mü']
        }
        
        self._flatten_suffixes()
        self._build_suffix_trie()
    
    def _flatten_suffixes(self):
        """Flatten suffixes into a single list sorted by length"""
        self.all_suffixes = []
        
        for category in self.suffixes.values():
            if isinstance(category, dict):
                for suffixes in category.values():
                    self.all_suffixes.extend(suffixes)
            else:
                self.all_suffixes.extend(category)
        
        # Remove duplicates and sort by length (longest first for greedy matching)
        self.all_suffixes = list(set(self.all_suffixes))
        self.all_suffixes.sort(key=len, reverse=True)
    
    def _build_suffix_trie(self):
        """Build a trie data structure for faster suffix matching"""
        self.suffix_trie = {}
        
        for suffix in self.all_suffixes:
            current = self.suffix_trie
            for char in reversed(suffix):  # Store suffixes in reverse for backward matching
                if char not in current:
                    current[char] = {}
                current = current[char]
            current['$'] = suffix  # Mark end of suffix
    
    def find_suffixes(self, word: str) -> Tuple[str, List[str]]:
        """
        Find possible suffixes in a Turkish word and return the stem and suffixes
        """
        word = word.lower()
        found_suffixes = []
        
        # Try to extract suffixes from the word
        stem = word
        while len(stem) > 2:  # Most Turkish stems are at least 2 chars
            found = False
            for suffix in self.all_suffixes:
                if stem.endswith(suffix) and len(stem) > len(suffix):
                    # Check if suffix is valid with vocal harmony
                    if self._check_vowel_harmony(stem[:-len(suffix)], suffix):
                        found_suffixes.insert(0, suffix)
                        stem = stem[:-len(suffix)]
                        found = True
                        break
            
            if not found:
                break
        
        return stem, found_suffixes
    
    def _get_last_vowel(self, text: str):
        """Get the last vowel in the text"""
        for char in reversed(text):
            if char in self.vowels:
                return char
        return None
    
    def _check_vowel_harmony(self, stem: str, suffix: str) -> bool:
        """Check if the suffix follows Turkish vowel harmony rules"""
        # Get the last vowel in the stem
        last_vowel = self._get_last_vowel(stem)
        
        # Get the first vowel in the suffix
        suffix_vowel = None
        for char in suffix:
            if char in self.vowels:
                suffix_vowel = char
                break
        
        # If either stem has no vowel or suffix has no vowel, harmony is valid
        if not last_vowel or not suffix_vowel:
            return True
        
        # Check vowel harmony
        if last_vowel in self.vowel_harmony_map:
            expected_vowels = self.vowel_harmony_map[last_vowel]
            return suffix_vowel in expected_vowels
        
        return False  # Unknown vowel in stem
    
    def segment_word(self, word: str) -> List[str]:
        """
        Segment a word into stem and suffixes for tokenization
        """
        stem, suffixes = self.find_suffixes(word)
        
        segments = [stem]
        segments.extend(suffixes)
        
        return segments


class ImprovedTurkishTokenizer:
    """
    Advanced tokenizer for Turkish language with morphological analysis
    """
    
    def __init__(self, vocab_size: int = 64000):  # Increased from 48000
        self.vocab_size = vocab_size
        self.vocab: Dict[str, int] = {}
        self.reverse_vocab: Dict[int, str] = {}
        
        # Special tokens
        self.special_tokens = {
            '<PAD>': 0,
            '<UNK>': 1,
            '<START>': 2,
            '<END>': 3,
            '<MASK>': 4,
            '<INSTRUCTION>': 5,
            '<OUTPUT>': 6,
            '<SEP>': 7,
            '<SPACE>': 8    # Add special token for space
        }
        
        self.special_tokens.update({
            '<UPPER>': 9,    # Büyük harf için özel token
            '<TITLE>': 10,   # Title case için özel token
            '<NUM_START>': 11,  # Sayı başlangıcı için
            '<NUM_END>': 12     # Sayı sonu için
        })
        
        # Case-specific special tokens ekleme
        self.special_tokens.update({
            '<UPPER_START>': 13,
            '<UPPER_END>': 14,
            '<TITLE_START>': 15,
            '<TITLE_END>': 16,
            '<TURKISH_I>': 17,
            '<TURKISH_i>': 18,
        })
        
        # Turkish alphabet and character sets
        self.tr_letters = set('abcçdefgğhıijklmnoöprsştuüvyzâîû')
        self.tr_vowels = set('aeıioöuüâîû')
        self.tr_consonants = set('bcçdfgğhjklmnprsştvyz')
        
        # Initialize morphological analyzer
        self.morphology = TurkishMorphology()
        
        # Counters for word and subword frequencies
        self.word_freq = Counter()
        self.subword_freq = Counter()
        self.bigram_scores = {}
        
        # Training metrics
        self.metrics = {
            'epoch_losses': [],
            'coverage_history': [],
            'vocab_growth': []
        }
        
        # Initialize vocabulary with special tokens
        self._initialize_vocab()
        
        # Metadata for the model
        self.metadata = {
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "vocab_size": vocab_size,
            "version": "2.1.0",
            "description": "Advanced Turkish-specific Tokenizer with Morphological Analysis"
        }
    
    def _initialize_vocab(self):
        """Initialize vocabulary with special tokens"""
        for token, idx in self.special_tokens.items():
            self.vocab[token] = idx
            self.reverse_vocab[idx] = token
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for consistent tokenization"""
        # Convert to lowercase
        text = text.lower()
        
        # Replace URLs, emails, and numbers with special tokens
        text = re.sub(r'https?://\S+|www\.\S+', '<URL>', text)
        text = re.sub(r'\S+@\S+\.\S+', '<EMAIL>', text)
        
        # Handle numbers
        text = re.sub(r'\d+[\.,]\d+', '<FLOAT>', text)
        text = re.sub(r'\d+', '<NUM>', text)
        
        # Handle common punctuation
        text = re.sub(r'[.]{3,}', '...', text)  # Normalize ellipsis
        
        # Normalize whitespace but preserve space information
        text = re.sub(r'\s+', ' ', text)
        
        # Case preservation iyileştirmesi
        text = text.replace("İ", "I").replace("i", "ı")  # Türkçe karakterler için özel işlem
        
        # Sayılar için daha iyi handling
        text = re.sub(r'(\d+)', r'<NUM_START>\1<NUM_END>', text)
        
        return text.strip()
    
    def _is_turkish_word(self, word: str) -> bool:
        """Check if a word is likely Turkish based on character composition"""
        # Remove non-letter characters
        word_clean = ''.join(c for c in word.lower() if c.isalpha())
        
        if not word_clean:
            return False
        
        # Check if word contains Turkish vowels and all characters are from Turkish alphabet
        has_turkish_vowel = any(c in self.tr_vowels for c in word_clean)
        all_turkish_chars = all(c in self.tr_letters for c in word_clean)
        
        # Check for consonant harmony which is common in Turkish
        consonant_count = sum(1 for c in word_clean if c in self.tr_consonants)
        vowel_count = sum(1 for c in word_clean if c in self.tr_vowels)
        
        # Most Turkish words don't have more than 2-3 consonants in a row
        consonant_sequence = max(len(seq) for seq in re.findall(r'[bcçdfgğhjklmnprsştvyz]+', word_clean) or [''])
        
        return (has_turkish_vowel and all_turkish_chars and 
                vowel_count > 0 and consonant_sequence <= 3)
    
    def _get_subwords(self, word: str, min_length: int = 2, max_length: int = 5) -> List[str]:
        """Extract potential subwords from a word"""
        subwords = []
        
        # 1. Character n-grams
        for n in range(min_length, min(max_length + 1, len(word) + 1)):
            for i in range(len(word) - n + 1):
                subword = word[i:i+n]
                subwords.append(subword)
        
        # 2. Add morphologically meaningful segments
        segments = self.morphology.segment_word(word)
        for segment in segments:
            if len(segment) >= min_length:
                subwords.append(segment)
        
        return list(set(subwords))  # Remove duplicates
    
    def _compute_bigram_scores(self, min_freq: int = 2):
        """Compute scores for adjacent character pairs based on frequency"""
        # Count character bigrams across the corpus
        bigram_counts = Counter()
        char_counts = Counter()
        
        # Process words that appear frequently enough
        for word, freq in self.word_freq.items():
            if freq >= min_freq and len(word) > 1:
                # Count characters
                for char in word:
                    char_counts[char] += freq
                
                # Count bigrams
                for i in range(len(word) - 1):
                    bigram = word[i:i+2]  # Get the bigram directly
                    bigram_counts[bigram] += freq
        
        # Calculate mutual information scores for bigrams
        total_chars = sum(char_counts.values())
        
        for bigram, count in bigram_counts.items():
            if count < min_freq:
                continue
            
            # Safely extract characters from bigram
            if len(bigram) == 2:
                char1, char2 = bigram[0], bigram[1]
                
                # Skip if component character counts are too low
                if char_counts[char1] < min_freq or char_counts[char2] < min_freq:
                    continue
                
                # Calculate pointwise mutual information
                p_xy = count / total_chars
                p_x = char_counts[char1] / total_chars
                p_y = char_counts[char2] / total_chars
                
                try:
                    pmi = np.log2(p_xy / (p_x * p_y))
                    self.bigram_scores[bigram] = pmi
                except (ZeroDivisionError, ValueError) as e:
                    print(f"Warning: PMI calculation failed for bigram {bigram}: {e}")
                    continue
    
    def _get_merge_candidates(self) -> List[Tuple[str, str, float]]:
        """Get candidate pairs for merging based on frequency and cohesion"""
        candidates = []
        
        # Find adjacent pairs that occur frequently together
        adjacent_pairs = Counter()
        
        for word, freq in self.word_freq.items():
            if freq < 3 or len(word) < 2:
                continue
                
            # Get tokens for this word
            tokens = []
            for char in word:
                if char in self.vocab:
                    tokens.append(char)
                else:
                    tokens.append('<UNK>')
            
            # Count adjacent pairs
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i+1])
                adjacent_pairs[pair] += freq
        
        # Convert to list of (token1, token2, score) tuples
        for (token1, token2), freq in adjacent_pairs.most_common(5000):
            # Skip if resulting merge would exceed max token length
            if len(token1) + len(token2) > 15:
                continue
                
            # Calculate score based on frequency and bigram cohesion
            score = freq
            
            # Add bigram statistics if available
            bigram = token1[-1] + token2[0] if token1 and token2 else ""
            if bigram in self.bigram_scores:
                # Boost score for high PMI bigrams
                score *= (1 + max(0, self.bigram_scores[bigram]))
            
            candidates.append((token1, token2, score))
        
        return sorted(candidates, key=lambda x: x[2], reverse=True)
    
    def tokenize(self, text: str) -> List[int]:
        """
        Tokenize text into token IDs with improved case preservation
        """
        if not text:
            return [self.special_tokens['<START>'], self.special_tokens['<END>']]
        
        # Save original casing information
        original_case = {}
        for i, char in enumerate(text):
            if char.isupper():
                original_case[i] = 'upper'
            elif char.islower():
                original_case[i] = 'lower'
        
        # Store words that should always be capitalized
        preserved_words = {}
        words = text.split()
        char_position = 0
        
        for word in words:
            if word.isupper() and len(word) >= 2:  # All caps words like "STEM"
                preserved_words[char_position] = word
            elif word[0].isupper() and len(word) > 1:  # Proper nouns or start of sentence
                preserved_words[char_position] = word
            char_position += len(word) + 1  # +1 for the space
        
        # Normalize text for consistent tokenization while saving metadata
        text = self._normalize_text(text)
        tokens = [self.special_tokens['<START>']]
        
        # Store the original case info in the tokenizer object so it can be accessed in decode
        self._last_original_case = original_case
        self._last_preserved_words = preserved_words
        
        # Process text word by word with explicit space handling
        words = text.split()
        
        case_info = []
        for word in text.split():
            if word.isupper():
                case_info.append(('<UPPER>', word))
            elif word.istitle():
                case_info.append(('<TITLE>', word))
        
        for i, word in enumerate(words):
            # Handle special markers
            if word == '|':
                tokens.append(self.special_tokens['<SEP>'])
                continue
            elif word.startswith('soru:'):
                tokens.append(self.special_tokens['<INSTRUCTION>'])
                word = word[5:].strip()
            elif word.startswith('cevap:'):
                tokens.append(self.special_tokens['<OUTPUT>'])
                word = word[6:].strip()
            
            # First try the whole word
            if word in self.vocab:
                tokens.append(self.vocab[word])
            else:
                # Use morphological segmentation for Turkish words
                if self._is_turkish_word(word):
                    segments = self.morphology.segment_word(word)
                    
                    # Try to tokenize each segment
                    segment_tokens = []
                    for segment in segments:
                        if segment in self.vocab:
                            segment_tokens.append(self.vocab[segment])
                        else:
                            # Try character by character for unknown segments
                            for char in segment:
                                if char in self.vocab:
                                    segment_tokens.append(self.vocab[char])
                                else:
                                    segment_tokens.append(self.special_tokens['<UNK>'])
                    
                    tokens.extend(segment_tokens)
                else:
                    # For non-Turkish words, try character-level tokenization
                    for char in word:
                        if char in self.vocab:
                            tokens.append(self.vocab[char])
                        else:
                            tokens.append(self.special_tokens['<UNK>'])
            
            # Add space token after each word except the last one
            if i < len(words) - 1:
                tokens.append(self.special_tokens['<SPACE>'])
        
        tokens.append(self.special_tokens['<END>'])
        return tokens
    
    def decode(self, token_ids: List[int]) -> str:
        """
        Convert token IDs back to text with proper word spacing and case preservation
        """
        result = []
        buffer = []
        capitalize_next = True  # For sentence start capitalization
        capitalize_all_next = False  # For cases like "STEM" where everything is uppercase
        is_instruction_part = False
        is_output_part = False
        first_word_in_segment = True
        
        for idx in token_ids:
            if idx not in self.reverse_vocab:
                continue
                
            token = self.reverse_vocab[idx]
            
            # Check if it's a special token by ID
            is_special = idx in self.special_tokens.values()
            special_name = None
            if is_special:
                # Get the special token name
                for name, token_id in self.special_tokens.items():
                    if token_id == idx:
                        special_name = name
                        break
            
            # Handle special tokens
            if is_special:
                if special_name == '<SPACE>':
                    # Handle explicit space tokens
                    if buffer:
                        word = ''.join(buffer)
                        # Apply capitalization
                        if capitalize_all_next:
                            word = word.upper()
                            capitalize_all_next = False
                        elif capitalize_next and word:
                            word = word[0].upper() + word[1:] if len(word) > 1 else word.upper()
                        result.append(word)
                        buffer = []
                        capitalize_next = False
                        first_word_in_segment = False
                    result.append(' ')
                elif special_name == '<START>':
                    # Start token - next word should be capitalized
                    capitalize_next = True
                    first_word_in_segment = True
                    continue
                elif special_name == '<END>' or special_name == '<PAD>':
                    # Flush buffer and skip these tokens
                    if buffer:
                        word = ''.join(buffer)
                        if capitalize_all_next:
                            word = word.upper()
                        elif capitalize_next and word:
                            word = word[0].upper() + word[1:] if len(word) > 1 else word.upper()
                        result.append(word)
                        buffer = []
                    continue
                elif special_name == '<UNK>':
                    buffer.append('�')
                elif special_name == '<MASK>':
                    buffer.append('[MASK]')
                elif special_name == '<INSTRUCTION>':
                    # Flush any pending words
                    if buffer:
                        word = ''.join(buffer)
                        if capitalize_next and word:
                            word = word[0].upper() + word[1:] if len(word) > 1 else word.upper()
                        result.append(word)
                        buffer = []
                    
                    # Add space if needed before "Soru:"
                    if result and result[-1] != ' ':
                        result.append(' ')
                    
                    # Add "Soru:" formatted properly
                    result.append('Soru:')
                    is_instruction_part = True
                    is_output_part = False
                    capitalize_next = True
                    first_word_in_segment = True
                    continue
                elif special_name == '<OUTPUT>':
                    # Flush any pending words
                    if buffer:
                        word = ''.join(buffer)
                        if capitalize_next and word:
                            word = word[0].upper() + word[1:] if len(word) > 1 else word.upper()
                        result.append(word)
                        buffer = []
                    
                    # Add space if needed before "Cevap:"
                    if result and result[-1] != ' ':
                        result.append(' ')
                    
                    # Add "Cevap:" formatted properly
                    result.append('Cevap:')
                    is_instruction_part = False
                    is_output_part = True
                    capitalize_next = True
                    first_word_in_segment = True
                    continue
                elif special_name == '<SEP>':
                    # Flush any pending words
                    if buffer:
                        word = ''.join(buffer)
                        if capitalize_next and word:
                            word = word[0].upper() + word[1:] if len(word) > 1 else word.upper()
                        result.append(word)
                        buffer = []
                    
                    # Add separator with proper spacing
                    result.append(' | ')
                    capitalize_next = True
                    first_word_in_segment = True
                    continue
            else:
                # Handle special cases explicitly
                if token == '<NUM>':
                    buffer.append('5')  # Default number replacement
                elif token == '<FLOAT>':
                    buffer.append('5.0')
                elif token == '<URL>':
                    buffer.append('https://example.com')
                elif token == '<EMAIL>':
                    buffer.append('example@example.com')
                else:
                    # Regular tokens
                    # Check if token should trigger capitalization for next token
                    if token in ['.', '!', '?'] and buffer:
                        capitalize_next = True
                    
                    # Check for all caps words like "STEM" (4 or fewer chars, all caps)
                    if token.isupper() and len(token) <= 4:
                        capitalize_all_next = True
                    
                    # Check if in a special segment and this is a keyword
                    if first_word_in_segment:
                        if token == 'stem':
                            # Common acronyms that should be uppercase
                            buffer.append('STEM')
                            first_word_in_segment = False
                            continue
                        # More special cases can be added here
                    
                    # For normal words, preserve original capitalization when possible
                    buffer.append(token)
                    
                    # Since we processed a token, we're no longer at the first word
                    if token not in [',', '.', ';', ':', '"', "'", '(', ')', '[', ']']:
                        first_word_in_segment = False
        
        # Flush any remaining tokens in buffer
        if buffer:
            word = ''.join(buffer)
            if capitalize_all_next:
                word = word.upper()
            elif capitalize_next and word:
                word = word[0].upper() + word[1:] if len(word) > 1 else word.upper()
            result.append(word)
        
        # Join with proper spacing
        text = ''
        for i, part in enumerate(result):
            if part == ' ':
                text += part
            elif i > 0 and result[i-1] != ' ' and not part.startswith(('|', '.', ',', '!', '?', ':', ';', '"', "'", '(', ')', '[', ']')):
                text += ' ' + part
            else:
                text += part
        
        # Final cleanup and formatting
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        text = re.sub(r'\s+([.,;:!?)])', r'\1', text)  # Remove spaces before punctuation
        text = re.sub(r'(\()\s+', r'\1', text)  # Remove spaces after opening parenthesis
        
        # Fix common capitalization patterns
        text = re.sub(r'\bi̇\b', 'İ', text)  # Properly capitalize dotted İ
        
        # Ensure "Soru:" and "Cevap:" are properly capitalized
        text = text.replace("soru:", "Soru:")
        text = text.replace("cevap:", "Cevap:")
        
        # Fix spacing around special markers
        text = text.replace("Soru: ", "Soru: ")
        text = text.replace("Cevap: ", "Cevap: ")
        text = text.replace(" | ", " | ")
        
        # Prevent double spaces
        while '  ' in text:
            text = text.replace('  ', ' ')
        
        return text.strip()

    def train(self, texts: List[str], epochs: int = 15,  # Increased from 10
              min_freq: int = 3, learning_rate: float = 0.003,  # Reduced for stability
              save_path: Optional[str] = None, 
              eval_texts: Optional[List[str]] = None,
              patience: int = 5):  # Patience artırıldı
        """
        Train the tokenizer on the provided texts with multiple epochs
        
        Args:
            texts: List of training texts
            epochs: Number of training epochs
            min_freq: Minimum frequency for a token to be included
            learning_rate: Learning rate for training
            save_path: Directory to save the model
            eval_texts: Optional evaluation texts
        """
        print(f"Tokenizer eğitimi başlatılıyor - Hedef sözlük boyutu: {self.vocab_size}")
        print(f"Eğitim paramterleri: {epochs} epoch, min_freq={min_freq}, lr={learning_rate}")
        start_time = time.time()
        
        # Early stopping için değişkenler
        best_coverage = 0.0
        patience_counter = 0
        best_epoch = 0
        min_improvement = 0.0005  # Daha hassas improvement check
        no_improvement_epochs = 0
        
        # Initial vocabulary with characters and special tokens
        if len(self.vocab) <= len(self.special_tokens):
            # Add all Turkish characters to the vocabulary
            vocab_idx = len(self.special_tokens)
            for char in 'abcçdefgğhıijklmnoöprsştuüvyzâîû0123456789.,!?\'"-:;()[]{}':
                if char not in self.vocab:
                    self.vocab[char] = vocab_idx
                    self.reverse_vocab[vocab_idx] = char
                    vocab_idx += 1
        
        # First pass: collect word and character statistics
        print("İlk geçiş: Kelime istatistikleri hesaplanıyor...")
        for text in tqdm(texts, desc="Metinler işleniyor"):
            text = self._normalize_text(text)
            
            for word in text.split():
                # Skip special markers
                if word in ('|', 'soru:', 'cevap:'):
                    continue
                    
                # Count whole words
                self.word_freq[word] += 1
                
                # Count characters
                for char in word:
                    if char not in self.vocab and vocab_idx < self.vocab_size:
                        self.vocab[char] = vocab_idx
                        self.reverse_vocab[vocab_idx] = char
                        vocab_idx += 1
        
        # Compute bigram statistics for better merges
        print("İkili karakter istatistikleri hesaplanıyor...")
        self._compute_bigram_scores(min_freq=min_freq)
        
        # Multi-epoch training
        print(f"\n{epochs} epoch için eğitim başlatılıyor...")
        
        # Training metrics
        best_coverage = 0.0
        
        for epoch in range(1, epochs + 1):
            epoch_start = time.time()
            print(f"\nEpoch {epoch}/{epochs}")
            
            # Collect subword statistics
            self.subword_freq = Counter()
            
            # Process frequent words to extract subwords
            for word, freq in tqdm(self.word_freq.most_common(10000), 
                                 desc=f"Morfolojik analiz (Epoch {epoch})"):
                if freq < min_freq:
                    continue
                    
                if self._is_turkish_word(word):
                    subwords = self._get_subwords(word)
                    for subword in subwords:
                        self.subword_freq[subword] += freq
            
            # Update vocabulary with common subwords
            print(f"Sözlük güncelleniyor (Epoch {epoch})...")
            
            # Track vocabulary size before update
            prev_vocab_size = len(self.vocab)
            
            # Add common words to vocabulary
            for word, freq in self.word_freq.most_common():
                if len(self.vocab) >= self.vocab_size - 1000:  # Reserve some space for subwords
                    break
                    
                if freq >= min_freq * 2 and word not in self.vocab:  # Higher threshold for whole words
                    self.vocab[word] = len(self.vocab)
                    self.reverse_vocab[len(self.vocab) - 1] = word
            
            # Add common subwords to vocabulary
            for subword, freq in self.subword_freq.most_common():
                if len(self.vocab) >= self.vocab_size:
                    break
                    
                if freq >= min_freq and subword not in self.vocab:
                    self.vocab[subword] = len(self.vocab)
                    self.reverse_vocab[len(self.vocab) - 1] = subword
            
            # Track vocabulary growth
            self.metrics['vocab_growth'].append(len(self.vocab))
            
            # Add merged tokens based on frequency and cohesion
            if epoch > 1:
                merge_candidates = self._get_merge_candidates()
                merges_added = 0
                
                for token1, token2, score in merge_candidates:
                    if len(self.vocab) >= self.vocab_size:
                        break
                        
                    merged = token1 + token2
                    if merged not in self.vocab:
                        self.vocab[merged] = len(self.vocab)
                        self.reverse_vocab[len(self.vocab) - 1] = merged
                        merges_added += 1
                        
                        if merges_added >= int((self.vocab_size - prev_vocab_size) * 0.3):
                            break
                
                print(f"Epoch {epoch}: {merges_added} birleştirilmiş token eklendi")
            
            # Evaluate on training data
            if eval_texts:
                coverage = self._evaluate_epoch(eval_texts)
                
                # Early stopping kontrolü
                if coverage > best_coverage + min_improvement:  # Daha hassas improvement check
                    best_coverage = coverage
                    patience_counter = 0
                    best_epoch = epoch
                    if save_path:
                        best_model_path = f"{save_path}_best"
                        self.save(best_model_path)
                        print(f"  ✓ En iyi model kaydedildi (epoch {epoch}): {best_model_path}")
                else:
                    patience_counter += 1
                    print(f"  ! İyileşme yok - Patience: {patience_counter}/{patience}")
                    
                if patience_counter >= patience:
                    print(f"\nEarly stopping! {patience} epoch boyunca iyileşme olmadı.")
                    print(f"En iyi sonuç epoch {best_epoch}'de alındı: {best_coverage:.2f}%")
                    break
            
            # Adaptive learning rate
            if epoch > 1 and epoch % 3 == 0:
                learning_rate *= 0.8  # Reduce learning rate every 3 epochs
                print(f"Learning rate güncellendi: {learning_rate:.6f}")
            
            # Evaluate on training data
            if eval_texts:
                eval_set = eval_texts
            else:
                # Use a sample of training data if no eval set provided
                eval_set = random.sample(texts, min(100, len(texts)))
            
            total_tokens = 0
            unknown_tokens = 0
            
            for eval_text in tqdm(eval_set, desc=f"Değerlendirme (Epoch {epoch})"):
                tokens = self.tokenize(eval_text)
                total_tokens += len(tokens)
                unknown_tokens += tokens.count(self.special_tokens['<UNK>'])
            
            # Calculate coverage
            coverage = 100.0 * (total_tokens - unknown_tokens) / total_tokens
            self.metrics['coverage_history'].append(coverage)
            
            print(f"Epoch {epoch} değerlendirme sonuçları:")
            print(f"  Sözlük boyutu: {len(self.vocab)}/{self.vocab_size}")
            print(f"  Token kapsama oranı: {coverage:.2f}%")
            print(f"  Epoch süresi: {time.time() - epoch_start:.2f} saniye")
            
            # Save best model
            if coverage > best_coverage:
                best_coverage = coverage
                if save_path:
                    best_model_path = f"{save_path}_best"
                    self.save(best_model_path)
                    print(f"  ✓ En iyi model kaydedildi: {best_model_path}")
        
        # Final stats
        train_time = time.time() - start_time
        vocab_coverage = len(self.vocab) / self.vocab_size * 100
        
        print(f"\nEğitim tamamlandı!")
        print(f"Toplam süre: {train_time:.2f} saniye")
        print(f"Son sözlük boyutu: {len(self.vocab)}/{self.vocab_size} ({vocab_coverage:.1f}%)")
        print(f"Son token kapsama oranı: {self.metrics['coverage_history'][-1]:.2f}%")
        
        # Save final model if requested
        if save_path:
            self.save(save_path)
            
            # Create and save visualizations
            self._create_training_visualizations(save_path)
    
    def _evaluate_epoch(self, eval_texts: List[str]) -> float:
        """Tek epoch için değerlendirme"""
        total_tokens = 0
        unknown_tokens = 0
        
        # Random sample for large eval sets
        if len(eval_texts) > 1000:
            eval_sample = random.sample(eval_texts, 1000)
        else:
            eval_sample = eval_texts
            
        for text in tqdm(eval_sample, desc="Değerlendirme"):
            tokens = self.tokenize(text)
            total_tokens += len(tokens)
            unknown_tokens += tokens.count(self.special_tokens['<UNK>'])
        
        coverage = 100.0 * (total_tokens - unknown_tokens) / total_tokens
        return coverage

    def _create_training_visualizations(self, save_path: str):
        """Create and save training visualizations"""
        try:
            plt.figure(figsize=(12, 8))
            
            # Plot coverage history
            plt.subplot(2, 1, 1)
            plt.plot(range(1, len(self.metrics['coverage_history']) + 1), 
                    self.metrics['coverage_history'], 'b-o')
            plt.title('Eğitim Boyunca Token Kapsama Oranı')
            plt.xlabel('Epoch')
            plt.ylabel('Kapsama Oranı (%)')
            plt.grid(True)
            
            # Plot vocabulary growth
            plt.subplot(2, 1, 2)
            plt.plot(range(1, len(self.metrics['vocab_growth']) + 1), 
                    self.metrics['vocab_growth'], 'g-o')
            plt.title('Sözlük Boyutu Gelişimi')
            plt.xlabel('Epoch')
            plt.ylabel('Sözlük Boyutu')
            plt.grid(True)
            
            plt.tight_layout()
            
            # Save figure
            plt.savefig(f"{save_path}/training_metrics.png")
            print(f"Eğitim metrikleri görselleştirildi: {save_path}/training_metrics.png")
            
            # Create token embedding visualization for a sample of tokens
            if len(self.vocab) > 20:
                self._visualize_token_embeddings(save_path)
                
        except Exception as e:
            print(f"Görselleştirme oluşturulurken hata: {e}")
    
    def _visualize_token_embeddings(self, save_path: str, sample_size: int = 200):
        """Create t-SNE visualization of token relationships"""
        try:
            # Create simple character-based feature vectors for tokens
            tokens = list(self.vocab.keys())
            
            # Take a sample of tokens, excluding special tokens
            sample_tokens = [t for t in tokens if t not in self.special_tokens.keys()]
            if len(sample_tokens) > sample_size:
                sample_tokens = random.sample(sample_tokens, sample_size)
            
            # Create character-based feature vectors
            all_chars = set(''.join(sample_tokens))
            char_to_idx = {c: i for i, c in enumerate(all_chars)}
            
            # Create one-hot vectors based on character presence
            X = np.zeros((len(sample_tokens), len(all_chars)))
            for i, token in enumerate(sample_tokens):
                for char in token:
                    if char in char_to_idx:
                        X[i, char_to_idx[char]] = 1
            
            # Apply t-SNE for dimensionality reduction
            tsne = TSNE(n_components=2, random_state=42)
            X_tsne = tsne.fit_transform(X)
            
            # Plot results
            plt.figure(figsize=(12, 10))
            plt.scatter(X_tsne[:, 0], X_tsne[:, 1], alpha=0.6)
            
            # Add labels for a subset of tokens
            for i, token in enumerate(sample_tokens):
                if len(token) >= 3 and random.random() < 0.3:  # Only label some tokens to avoid clutter
                    plt.annotate(token, (X_tsne[i, 0], X_tsne[i, 1]), fontsize=8)
            
            plt.title('Token Embedding Visualization (t-SNE)')
            plt.tight_layout()
            plt.savefig(f"{save_path}/token_embeddings.png")
            print(f"Token görselleştirmesi kaydedildi: {save_path}/token_embeddings.png")
            
        except Exception as e:
            print(f"Token görselleştirmesi oluşturulurken hata: {e}")
    
    def evaluate(self, test_texts: List[str], num_examples: int = 5) -> Dict:
        """
        Evaluate the tokenizer on test texts with improved diagnostics
        
        Args:
            test_texts: List of texts for evaluation
            num_examples: Number of examples to print
        
        Returns:
            Dictionary with evaluation metrics
        """
        print("\nModel değerlendiriliyor...")
        
        total_tokens = 0
        unknown_tokens = 0
        original_chars = 0
        exact_matches = 0
        case_insensitive_matches = 0
        
        # Select random sample for example display
        display_indices = random.sample(range(len(test_texts)), min(num_examples, len(test_texts)))
        examples = []
        
        for i, test_text in enumerate(tqdm(test_texts, desc="Test ediliyor")):
            tokens = self.tokenize(test_text)
            decoded = self.decode(tokens)
            
            original_chars += len(test_text)
            total_tokens += len(tokens)
            unknown_tokens += tokens.count(self.special_tokens['<UNK>'])
            
            # Check for exact and semantic matches
            if decoded == test_text:
                exact_matches += 1
            elif decoded.lower() == test_text.lower():
                case_insensitive_matches += 1
            
            # Record example for display
            if i in display_indices:
                examples.append({
                    'original': test_text[:150] + ('...' if len(test_text) > 150 else ''),
                    'tokens': tokens[:20] + ['...'] if len(tokens) > 20 else tokens,
                    'token_count': len(tokens),
                    'decoded': decoded[:150] + ('...' if len(decoded) > 150 else ''),
                    'exact_match': decoded == test_text,
                    'case_match': decoded.lower() == test_text.lower()
                })
        
        # Calculate metrics
        coverage = (total_tokens - unknown_tokens) / total_tokens * 100 if total_tokens > 0 else 0
        compression_ratio = original_chars / total_tokens if total_tokens > 0 else 0
        exact_match_pct = exact_matches / len(test_texts) * 100 if test_texts else 0
        case_match_pct = case_insensitive_matches / len(test_texts) * 100 if test_texts else 0
        
        # Display examples
        for i, example in enumerate(examples):
            print(f"\nTest örneği {i+1}:")
            print(f"Orijinal: {example['original']}")
            print(f"Token sayısı: {example['token_count']}")
            token_preview = ' '.join([str(t) if isinstance(t, int) else t for t in example['tokens']])
            print(f"Tokenler: {token_preview}")
            print(f"Decode edilmiş: {example['decoded']}")
            print(f"Tam eşleşme: {'✓' if example['exact_match'] else '✗'}")
            if not example['exact_match'] and example['case_match']:
                print(f"Büyük-küçük harf eşleşmesi: ✓")
        
        # Display overall metrics
        print("\nDeğerlendirme Sonuçları:")
        print(f"Toplam token sayısı: {total_tokens}")
        print(f"Bilinmeyen token sayısı: {unknown_tokens}")
        print(f"Token kapsama oranı: {coverage:.2f}%")
        print(f"Sıkıştırma oranı: {compression_ratio:.2f} karakter/token")
        print(f"Tam eşleşme oranı: {exact_match_pct:.2f}%")
        print(f"Büyük-küçük harf duyarsız eşleşme: {case_match_pct:.2f}%")
        
        return {
            'total_tokens': total_tokens,
            'unknown_tokens': unknown_tokens,
            'coverage': coverage,
            'compression_ratio': compression_ratio,
            'exact_match_pct': exact_match_pct,
            'case_insensitive_match_pct': case_match_pct
        }
    
    def save(self, path: str):
        """Save the tokenizer model"""
        save_dir = Path(path)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Update metadata
        self.metadata["saved_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.metadata["vocab_size_actual"] = len(self.vocab)
        self.metadata["created_by"] = "emredeveloper"
        
        # Save vocabulary
        with open(save_dir / 'vocab.json', 'w', encoding='utf-8') as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=2)
        
        # Save frequencies
        with open(save_dir / 'frequencies.pkl', 'wb') as f:
            pickle.dump({
                'word_freq': self.word_freq,
                'subword_freq': self.subword_freq,
                'bigram_scores': self.bigram_scores
            }, f)
        
        # Save metadata
        with open(save_dir / 'metadata.json', 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)
        
        # Save metrics
        with open(save_dir / 'metrics.json', 'w', encoding='utf-8') as f:
            json.dump(self.metrics, f, ensure_ascii=False, indent=2)
        
        print(f"Model kaydedildi: {path}")

    @classmethod
    def load(cls, path: str) -> 'ImprovedTurkishTokenizer':
        """Load a tokenizer model from the given path"""
        load_dir = Path(path)
        
        if not load_dir.exists():
            raise ValueError(f"Model dizini bulunamadı: {path}")
        
        with open(load_dir / 'metadata.json', 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        tokenizer = cls(vocab_size=metadata['vocab_size'])
        tokenizer.metadata = metadata
        
        with open(load_dir / 'vocab.json', 'r', encoding='utf-8') as f:
            # Convert keys to strings as JSON always stores keys as strings
            vocab_data = json.load(f)
            tokenizer.vocab = {k: int(v) for k, v in vocab_data.items()}
            tokenizer.reverse_vocab = {int(v): k for k, v in vocab_data.items()}
        
        if (load_dir / 'frequencies.pkl').exists():
            with open(load_dir / 'frequencies.pkl', 'rb') as f:
                frequencies = pickle.load(f)
                tokenizer.word_freq = frequencies.get('word_freq', Counter())
                tokenizer.subword_freq = frequencies.get('subword_freq', Counter())
                tokenizer.bigram_scores = frequencies.get('bigram_scores', {})
        
        if (load_dir / 'metrics.json').exists():
            with open(load_dir / 'metrics.json', 'r', encoding='utf-8') as f:
                tokenizer.metrics = json.load(f)
        
        print(f"Model yüklendi: {path}")
        return tokenizer


def load_turkish_dataset(limit=5000):
    """Load the Turkish instructions dataset from Hugging Face"""
    try:
        print("Hugging Face'den veri seti yükleniyor...")
        dataset = load_dataset("NovusResearch/turkish_instructions")
        
        train_data = dataset['train']
        data = train_data.select(range(min(limit, len(train_data))))
        
        texts = []
        for item in tqdm(data, desc="Metinler işleniyor"):
            try:
                instruction = str(item['instruction']).strip()
                output = str(item['output']).strip()
                
                if instruction and output:
                    text = f"Soru: {instruction} | Cevap: {output}"
                    texts.append(text)
            except Exception as e:
                print(f"Veri işleme hatası: {e}")
                continue
        
        print(f"\nToplam {len(texts)} metin yüklendi.")
        
        if texts:
            print("\nVeri örnekleri:")
            for i, text in enumerate(texts[:3]):
                print(f"\nÖrnek {i+1}:")
                print(text[:150], "..." if len(text) > 150 else "")
        
        return texts
    except Exception as e:
        print(f"Veri yükleme hatası: {e}")
        import traceback
        traceback.print_exc()
        return []


def main():
    # Set random seed for reproducibility
    random.seed(42)
    
    # Load larger dataset
    texts = load_turkish_dataset(limit=20000)  # Increased from 10000
    
    if not texts:
        print("Veri yüklenemedi!")
        return
    
    # Better split ratios
    train_size = int(len(texts) * 0.7)  # Changed from 0.8
    val_size = int(len(texts) * 0.15)   # Changed from 0.1
    test_size = len(texts) - train_size - val_size
    
    train_texts = texts[:train_size]
    val_texts = texts[train_size:train_size + val_size]
    test_texts = texts[train_size + val_size:]
    
    print(f"\nVeri seti dağılımı:")
    print(f"Eğitim seti: {len(train_texts)} örnek ({len(train_texts)/len(texts)*100:.1f}%)")
    print(f"Doğrulama seti: {len(val_texts)} örnek ({len(val_texts)/len(texts)*100:.1f}%)")
    print(f"Test seti: {len(test_texts)} örnek ({len(test_texts)/len(texts)*100:.1f}%)")
    
    # Create tokenizer with increased vocab size
    vocab_size = 64000  # Increased from 48000
    tokenizer = ImprovedTurkishTokenizer(vocab_size=vocab_size)
    
    # Enhanced training parameters
    epochs = 15        # Increased from 10
    min_freq = 3
    learning_rate = 0.003  # Reduced for better stability
    patience = 3       # Early stopping patience
    
    # Create model directory with more descriptive name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = f"turkish_tokenizer_model_v3_{vocab_size}_{timestamp}"
    
    # Train with early stopping
    tokenizer.train(
        texts=train_texts,
        epochs=epochs, 
        min_freq=min_freq,
        learning_rate=learning_rate,
        save_path=model_dir,
        eval_texts=val_texts,
        patience=patience
    )
    
    # Comprehensive final evaluation
    print("\nKapsamlı final değerlendirmesi başlatılıyor...")
    tokenizer.evaluate(test_texts, num_examples=15)  # Increased from 10
    
    print(f"\nModel kaydedildi: {model_dir}")


if __name__ == "__main__":
    main()