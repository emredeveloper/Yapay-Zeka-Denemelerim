import time
from pathlib import Path
import json
from typing import List, Dict
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from train_tokenizer import ImprovedTurkishTokenizer, load_turkish_dataset
from tokenizer_utils import test_tokenizer_quality, batch_tokenize, batch_decode, find_latest_model

def run_benchmark(tokenizer_path: str, num_samples: int = 2000):  # Sample size artırıldı
    """Run comprehensive benchmark tests on the tokenizer"""
    print(f"Loading tokenizer from: {tokenizer_path}")
    tokenizer = ImprovedTurkishTokenizer.load(tokenizer_path)
    
    # Load test data
    print(f"\nLoading {num_samples} test samples...")
    test_texts = load_turkish_dataset(limit=num_samples)
    
    # 1. Quality Tests
    print("\nRunning quality tests...")
    quality_results = test_tokenizer_quality(tokenizer, test_texts)
    
    # 2. Performance Tests
    print("\nRunning performance tests...")
    perf_results = {
        "tokenization": [],
        "decoding": []
    }
    
    # Test different batch sizes
    batch_sizes = [1, 8, 16, 32, 64, 128, 256]  # Batch size aralığı genişletildi
    for batch_size in batch_sizes:
        # Tokenization speed
        start_time = time.time()
        tokens = batch_tokenize(tokenizer, test_texts[:100], batch_size=batch_size, show_progress=False)
        tok_time = time.time() - start_time
        
        # Decoding speed
        start_time = time.time()
        decoded = batch_decode(tokenizer, tokens, batch_size=batch_size, show_progress=False)
        dec_time = time.time() - start_time
        
        perf_results["tokenization"].append({
            "batch_size": batch_size,
            "time": tok_time,
            "tokens_per_second": len(test_texts[:100]) / tok_time
        })
        
        perf_results["decoding"].append({
            "batch_size": batch_size,
            "time": dec_time,
            "texts_per_second": len(tokens) / dec_time
        })
    
    # Initialize results dictionary first
    results = {
        "quality": quality_results,
        "performance": perf_results,
        "tokenizer_info": {
            "vocab_size": len(tokenizer.vocab),
            "special_tokens": len(tokenizer.special_tokens)
        },
        "special_cases": {}  # Initialize special_cases dict
    }
    
    # Case preservation testi eklendi
    special_cases = [
        "İSTANBUL'da YAPAY ZEKA",
        "Türkçe Doğal Dil İşleme",
        "Prof. Dr. Ahmet'in AI çalışması",
        "1. Bölüm: Test-123"
    ]
    
    # Add special case tests
    special_case_results = {}
    for case in special_cases:
        tokens = tokenizer.tokenize(case)
        decoded = tokenizer.decode(tokens)
        special_case_results[case] = {
            "original": case,
            "decoded": decoded,
            "match": case == decoded
        }
    
    results["special_cases"] = special_case_results
    
    # Save results
    output_path = Path("benchmark_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # Visualize results
    plot_benchmark_results(results)
    
    return results

def plot_benchmark_results(results: Dict):
    """Plot benchmark results"""
    plt.figure(figsize=(15, 10))
    
    # 1. Quality metrics
    plt.subplot(2, 2, 1)
    quality = results["quality"]
    metrics = ["exact_match_pct", "case_match_pct", "whitespace_match_pct"]
    plt.bar(range(len(metrics)), [quality[m] for m in metrics])
    plt.xticks(range(len(metrics)), ["Exact", "Case", "Whitespace"], rotation=45)
    plt.title("Quality Metrics (%)")
    
    # 2. Failure analysis
    plt.subplot(2, 2, 2)
    failures = quality["failure_types"]
    plt.pie(failures.values(), labels=failures.keys(), autopct='%1.1f%%')
    plt.title("Failure Analysis")
    
    # 3. Performance - Tokenization
    plt.subplot(2, 2, 3)
    tok_data = results["performance"]["tokenization"]
    plt.plot([d["batch_size"] for d in tok_data], 
            [d["tokens_per_second"] for d in tok_data], 
            marker='o')
    plt.title("Tokenization Speed")
    plt.xlabel("Batch Size")
    plt.ylabel("Tokens/second")
    
    # 4. Performance - Decoding
    plt.subplot(2, 2, 4)
    dec_data = results["performance"]["decoding"]
    plt.plot([d["batch_size"] for d in dec_data], 
            [d["texts_per_second"] for d in dec_data], 
            marker='o')
    plt.title("Decoding Speed")
    plt.xlabel("Batch Size")
    plt.ylabel("Texts/second")
    
    plt.tight_layout()
    plt.savefig("benchmark_results.png")
    print("\nBenchmark visualizations saved to benchmark_results.png")

if __name__ == "__main__":
    # Tokenizer'ı eğit
    from train_tokenizer import main as train_tokenizer
    print("Training tokenizer...")
    train_tokenizer()
    
    latest_model = find_latest_model()
    
    if latest_model:
        print(f"\nRunning benchmarks on {latest_model}")
        results = run_benchmark(latest_model)
        
        print("\nBenchmark Summary:")
        print(f"Exact Match Rate: {results['quality']['exact_match_pct']:.2f}%")
        print(f"Case Match Rate: {results['quality']['case_match_pct']:.2f}%")
        print(f"Vocab Size: {results['tokenizer_info']['vocab_size']}")
        print(f"Special Tokens: {results['tokenizer_info']['special_tokens']}")
    else:
        print("No tokenizer model found!")
