"""
Script to run the Enhanced Turkish Graph RAG system with the improved tokenizer.
This demonstrates better text preservation and case handling in the RAG pipeline.
"""
import os
import argparse
from pathlib import Path
import json
from datetime import datetime

# Import the RAG system and tokenizer utilities
from graph_rag import EnhancedTurkishGraphRAG
from train_tokenizer import ImprovedTurkishTokenizer, main as train_tokenizer, load_turkish_dataset
from tokenizer_utils import find_latest_model

def run_improved_rag(tokenizer_path: str = None, output_file: str = None):
    """
    Run the Graph RAG system using our improved tokenizer
    """
    # First check if we need to train a new model
    if not tokenizer_path or not Path(tokenizer_path).exists():
        print("No valid tokenizer model found. Training new model...")
        train_tokenizer()
        latest_model = find_latest_model()
        if not latest_model:
            raise ValueError("Failed to train tokenizer model!")
        tokenizer_path = latest_model

    print(f"Loading improved tokenizer from: {tokenizer_path}")
    tokenizer = ImprovedTurkishTokenizer.load(tokenizer_path)
    
    # Test documents with challenging cases for case preservation and formatting
    documents = [
        "Yapay Zeka, insan zekasını taklit eden ve öğrenebilen sistemlerdir.",
        "Derin Öğrenme, yapay sinir ağları kullanarak karmaşık örüntüleri tespit eder.",
        "Doğal Dil İşleme, bilgisayarların Türkçe dahil insan dillerini anlamasını sağlar.",
        "Makine Öğrenimi, verilerden öğrenerek kendini geliştiren algoritmaları kapsar.",
        "Bilgisayarlı Görü, görüntü ve videoları yapay zeka ile analiz eder.",
        "Türkçe Doğal Dil İşleme, Türkçe metinleri anlama ve işleme yeteneğidir.",
        "Türk dili, Ural-Altay dil ailesine mensup sondan eklemeli bir dildir.",
        "Transformatör mimarileri, NLP alanında çığır açan bir tekniktir.",
        "BERT modeli, çift yönlü kodlama yaparak bağlam analizini iyileştirir.",
        "GPT modelleri, ileriye doğru tahmin yaparak tutarlı metinler üretir.",
        "İstanbul, Türkiye'nin en büyük şehri ve kültürel başkentidir.",
        "TÜBİTAK tarafından desteklenen AI projeleri bilimsel gelişime katkı sağlar."
    ]
    
    # First test direct tokenizer case preservation on our documents
    print("\nTesting tokenizer case preservation on documents:")
    case_preservation_test = {
        "success_rate": 0,
        "examples": []
    }
    
    success_count = 0
    for i, doc in enumerate(documents):
        tokens = tokenizer.tokenize(doc)
        decoded = tokenizer.decode(tokens)
        
        is_match = decoded == doc
        if is_match:
            success_count += 1
            
        # Add example to our test results
        if i < 3 or not is_match:  # Always include first 3 and any failures
            case_preservation_test["examples"].append({
                "original": doc,
                "decoded": decoded,
                "match": "✓" if is_match else "✗"
            })
            
            print(f"\nDocument {i+1}:")
            print(f"Original: {doc}")
            print(f"Decoded:  {decoded}")
            print(f"Match:    {'✓' if is_match else '✗'}")
    
    # Calculate success rate
    case_preservation_test["success_rate"] = f"{success_count / len(documents) * 100:.1f}%"
    print(f"\nCase preservation success rate: {case_preservation_test['success_rate']}")
    
    # Create the RAG system with the improved tokenizer
    print(f"\nInitializing Graph RAG with improved tokenizer...")
    rag = EnhancedTurkishGraphRAG(use_tokenizer=True, tokenizer_path=tokenizer_path)
    
    # Build the knowledge graph
    print(f"Building knowledge graph from {len(documents)} documents...")
    rag.build_knowledge_graph(documents)
    
    # Run a series of test queries that exercise different aspects of the tokenizer
    queries = [
        "Yapay Zeka ve Doğal Dil İşleme arasındaki ilişki nedir?",
        "BERT ve GPT modelleri arasındaki farklar nelerdir?",
        "Türkçe dil işleme için hangi modeller kullanılır?",
        "İstanbul'un yapay zeka ekosistemi nasıldır?",
        "TÜBİTAK'ın NLP projelerine katkısı nedir?"
    ]
    
    # Process each query and collect results
    results = {}
    
    for i, query in enumerate(queries):
        print(f"\nProcessing query {i+1}: {query}")
        query_results = rag.query(query)
        
        print(f"Found {len(query_results)} relevant documents")
        
        # Format for display and storage
        results[f"query_{i+1}"] = {
            "query": query,
            "results": []
        }
        
        for j, result in enumerate(query_results[:3]):  # Top 3 results
            print(f"\nResult {j+1}:")
            print(f"Text: {result['text']}")
            print(f"Similarity: {result['similarity']:.3f}")
            print(f"Keywords: {', '.join(result['keywords'])}")
            
            # Add to results dict
            results[f"query_{i+1}"]["results"].append({
                "text": result['text'],
                "similarity": round(result['similarity'], 3),
                "importance": result['importance'],
                "keywords": result['keywords'],
                "cluster": result['cluster']
            })
    
    # Visualize the graph 
    print("\nVisualizing the knowledge graph...")
    graph_path = "knowledge_graph.png"
    rag.visualize(graph_path)
    print(f"Graph visualization saved to: {graph_path}")
    
    # Save the results to file if requested
    if output_file:
        output_data = {
            "metadata": {
                "tokenizer": tokenizer_path,
                "documents": len(documents),
                "queries": len(queries),
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            },
            "documents": documents,
            "case_preservation_test": case_preservation_test
        }
        
        # Add query results
        for i, query in enumerate(queries):
            output_data[f"query_{i+1}"] = results[f"query_{i+1}"]
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        print(f"\nResults saved to: {output_file}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Run Graph RAG with improved tokenizer")
    parser.add_argument('--tokenizer', type=str, default=None,
                        help='Path to the improved tokenizer model (optional)')
    parser.add_argument('--output', type=str, default="improved_rag_results.json",
                        help='Path to save the results')
    args = parser.parse_args()
    
    run_improved_rag(args.tokenizer, args.output)

if __name__ == "__main__":
    main()
