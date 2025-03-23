import os
from pathlib import Path
from typing import List, Dict, Optional
from train_tokenizer import ImprovedTurkishTokenizer
from tqdm.auto import tqdm

def find_latest_model(base_dir: str = "./") -> Optional[str]:
    """Find the most recently created tokenizer model in the given directory"""
    models = [d for d in os.listdir(base_dir) 
              if os.path.isdir(os.path.join(base_dir, d)) 
              and d.startswith("turkish_tokenizer_model_")]
    
    if not models:
        return None
        
    # Sort by creation time (newest first)
    models.sort(key=lambda d: os.path.getctime(os.path.join(base_dir, d)), reverse=True)
    return os.path.join(base_dir, models[0])

def test_tokenizer_quality(tokenizer: ImprovedTurkishTokenizer, test_texts: List[str]) -> Dict:
    """Test the quality of tokenization"""
    total_tokens = 0
    unknown_tokens = 0
    exact_matches = 0
    case_matches = 0
    whitespace_matches = 0
    failure_types = {
        'case_issues': 0,
        'whitespace_issues': 0,
        'punctuation_issues': 0,
        'special_token_issues': 0,
        'character_changes': 0
    }
    
    for text in tqdm(test_texts, desc="Testing tokenizer quality"):
        if not text.strip():
            continue
            
        tokens = tokenizer.tokenize(text)
        decoded = tokenizer.decode(tokens)
        
        total_tokens += len(tokens)
        unknown_tokens += tokens.count(tokenizer.special_tokens['<UNK>'])
        
        if decoded == text:
            exact_matches += 1
        elif decoded.lower() == text.lower():
            case_matches += 1
            failure_types['case_issues'] += 1
        elif ''.join(decoded.split()) == ''.join(text.split()):
            whitespace_matches += 1
            failure_types['whitespace_issues'] += 1
        else:
            failure_types['character_changes'] += 1
    
    return {
        'total_tokens': total_tokens,
        'unknown_tokens': unknown_tokens,
        'exact_match_pct': (exact_matches / len(test_texts)) * 100,
        'case_match_pct': (case_matches / len(test_texts)) * 100,
        'whitespace_match_pct': (whitespace_matches / len(test_texts)) * 100,
        'failure_types': failure_types
    }

def batch_tokenize(tokenizer: ImprovedTurkishTokenizer, texts: List[str], 
                  batch_size: int = 64, show_progress: bool = True) -> List[List[int]]:
    """Tokenize a batch of texts"""
    results = []
    iterator = range(0, len(texts), batch_size)
    if show_progress:
        iterator = tqdm(iterator, desc="Tokenizing")
    
    for i in iterator:
        batch = texts[i:i+batch_size]
        batch_tokens = [tokenizer.tokenize(text) for text in batch]
        results.extend(batch_tokens)
    
    return results

def batch_decode(tokenizer: ImprovedTurkishTokenizer, token_lists: List[List[int]], 
                batch_size: int = 64, show_progress: bool = True) -> List[str]:
    """Decode a batch of token lists"""
    results = []
    iterator = range(0, len(token_lists), batch_size)
    if show_progress:
        iterator = tqdm(iterator, desc="Decoding")
    
    for i in iterator:
        batch = token_lists[i:i+batch_size]
        batch_texts = [tokenizer.decode(tokens) for tokens in batch]
        results.extend(batch_texts)
    
    return results
