from typing import List, Dict, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx

class TurkishTextProcessor:
    def __init__(self, use_tokenizer: bool = True, tokenizer_path: Optional[str] = None):
        self.use_tokenizer = use_tokenizer
        self.tokenizer = None
        if use_tokenizer:
            from train_tokenizer import ImprovedTurkishTokenizer  # Changed from improved_tokenizer
            self.tokenizer = ImprovedTurkishTokenizer.load(tokenizer_path) if tokenizer_path else None
        self.vectorizer = TfidfVectorizer(lowercase=True, max_features=5000)
    
    def preprocess_text(self, text: str) -> str:
        if self.tokenizer:
            # Case preservation için özel işlem
            case_preserved = []
            for word in text.split():
                if word.isupper():
                    case_preserved.append(f"<UPPER>{word}</UPPER>")
                elif word.istitle():
                    case_preserved.append(f"<TITLE>{word}</TITLE>")
                else:
                    case_preserved.append(word)
            text = " ".join(case_preserved)
            
            tokens = self.tokenizer.tokenize(text)
            return self.tokenizer.decode(tokens)
            
        return text.lower()
    
    def extract_keywords(self, text: str, top_k: int = 5) -> List[str]:
        # Basit keyword extraction
        words = text.lower().split()
        word_freq = {}
        for word in words:
            if len(word) > 2:  # En az 3 karakterli kelimeler
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # En sık geçen kelimeleri al
        keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, _ in keywords[:top_k]]
    
    def rank_documents(self, query: str, documents: List[str]) -> List[Dict]:
        # TF-IDF vektörleri oluştur
        all_texts = [query] + documents
        tfidf_matrix = self.vectorizer.fit_transform(all_texts)
        
        # Benzerlik hesapla
        similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])
        
        # Sonuçları sırala
        ranked = []
        for idx, score in enumerate(similarities[0]):
            ranked.append({
                'id': idx,
                'similarity': float(score)
            })
        
        return sorted(ranked, key=lambda x: x['similarity'], reverse=True)

class TurkishGraphAnalyzer:
    def __init__(self, text_processor: TurkishTextProcessor):
        self.text_processor = text_processor
    
    def find_document_clusters(self, documents: List[str], n_clusters: int = 3) -> List[int]:
        if len(documents) < n_clusters:
            return list(range(len(documents)))
        
        # TF-IDF vektörleri oluştur
        tfidf_matrix = self.text_processor.vectorizer.fit_transform(documents)
        
        # Kümeleme yap
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(tfidf_matrix.toarray())
        
        return clusters.tolist()
    
    def compute_edge_weights(self, doc1: str, doc2: str) -> float:
        # İki doküman arasındaki benzerliği hesapla
        tfidf = self.text_processor.vectorizer.fit_transform([doc1, doc2])
        similarity = cosine_similarity(tfidf[0:1], tfidf[1:])
        return float(similarity[0][0])
    
    def analyze_graph_structure(self, graph: nx.DiGraph) -> Dict:
        metrics = {
            'node_count': graph.number_of_nodes(),
            'edge_count': graph.number_of_edges(),
            'density': nx.density(graph),
            'avg_clustering': nx.average_clustering(graph),
            'connected_components': nx.number_connected_components(graph.to_undirected())
        }
        
        try:
            metrics['avg_shortest_path'] = nx.average_shortest_path_length(graph)
        except:
            metrics['avg_shortest_path'] = 'N/A'
        
        return metrics
