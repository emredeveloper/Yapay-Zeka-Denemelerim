from train_tokenizer import ImprovedTurkishTokenizer  # Changed from improved_tokenizer
from typing import List, Dict, Set, Optional
import networkx as nx  # import eklendi
import matplotlib.pyplot as plt  # import eklendi
from utils import TurkishTextProcessor, TurkishGraphAnalyzer
import os
from pathlib import Path

class EnhancedTurkishGraphRAG:
    def __init__(self, use_tokenizer: bool = True, tokenizer_path: Optional[str] = None):
        """
        Initialize the Enhanced Turkish Graph RAG system
        
        Args:
            use_tokenizer: Whether to use the Turkish tokenizer
            tokenizer_path: Path to tokenizer model (uses latest if None)
        """
        self.graph = nx.DiGraph()
        self.text_processor = TurkishTextProcessor(use_tokenizer=use_tokenizer, 
                                                 tokenizer_path=tokenizer_path)
        self.graph_analyzer = TurkishGraphAnalyzer(text_processor=self.text_processor)
        self.documents = []
        
    def build_knowledge_graph(self, documents: List[str]):
        """
        Build an enhanced knowledge graph from documents
        
        Args:
            documents: List of text documents
        """
        self.documents = documents
        
        # Preprocess documents
        processed_docs = [self.text_processor.preprocess_text(doc) for doc in documents]
        
        # Create nodes for each document
        for i, (doc, proc_doc) in enumerate(zip(documents, processed_docs)):
            keywords = self.text_processor.extract_keywords(proc_doc)
            
            # Create token-based representation if tokenizer is available
            token_ids = []
            if self.text_processor.use_tokenizer and self.text_processor.tokenizer:
                try:
                    token_ids = self.text_processor.tokenizer.tokenize(proc_doc)
                except Exception as e:
                    print(f"Warning: Tokenization failed for document {i}: {e}")
            
            # Add node with document data
            self.graph.add_node(i, 
                              text=doc,
                              processed_text=proc_doc,
                              keywords=keywords,
                              token_ids=token_ids)
        
        # Find document clusters
        clusters = self.graph_analyzer.find_document_clusters(processed_docs)
        
        # Add cluster info to nodes
        if clusters and len(clusters) == len(documents):
            for i, cluster_id in enumerate(clusters):
                self.graph.nodes[i]['cluster'] = cluster_id
        
        # Create edges based on document similarity
        for i in range(len(documents)):
            for j in range(i + 1, len(documents)):
                weight = self.graph_analyzer.compute_edge_weights(
                    processed_docs[i], processed_docs[j]
                )
                if weight > 0.3:  # Reduced threshold for more connections
                    self.graph.add_edge(i, j, weight=weight)
                    self.graph.add_edge(j, i, weight=weight)  # Make bidirectional

    def query(self, question: str, top_k: int = 3):
        """
        Process a query against the knowledge graph
        
        Args:
            question: Query text
            top_k: Number of top results to return
            
        Returns:
            List of relevant document results
        """
        # Preprocess query
        processed_question = self.text_processor.preprocess_text(question)
        
        # Get token-based representation if tokenizer is available
        query_tokens = []
        if self.text_processor.use_tokenizer and self.text_processor.tokenizer:
            try:
                query_tokens = self.text_processor.tokenizer.tokenize(processed_question)
            except Exception as e:
                print(f"Warning: Query tokenization failed: {e}")
        
        # Rank documents by relevance
        ranked_docs = self.text_processor.rank_documents(
            processed_question, 
            [self.graph.nodes[i]['processed_text'] for i in range(len(self.documents))]
        )
        
        # Expand context using graph connections
        context_nodes = set()
        
        # Start with top-ranked documents
        for doc in ranked_docs[:top_k]:
            node_id = doc['id']
            context_nodes.add(node_id)
            
            # Add neighbors with weight above threshold
            for neighbor in self.graph.neighbors(node_id):
                edge_data = self.graph.get_edge_data(node_id, neighbor)
                if edge_data and edge_data.get('weight', 0) > 0.4:  # Higher threshold for neighbors
                    context_nodes.add(neighbor)
        
        # Add any nodes that share tokens with the query (if tokenizer available)
        if query_tokens:
            for i in range(len(self.documents)):
                node_tokens = self.graph.nodes[i].get('token_ids', [])
                if node_tokens:
                    # Check for token overlap
                    common_tokens = set(query_tokens).intersection(set(node_tokens))
                    if len(common_tokens) >= 3:  # At least 3 common tokens
                        context_nodes.add(i)
        
        # Prepare results with enhanced context
        results = []
        for node_id in context_nodes:
            node_data = self.graph.nodes[node_id]
            
            # Get similarity score from ranked_docs or calculate directly
            similarity = next((doc['similarity'] 
                             for doc in ranked_docs 
                             if doc['id'] == node_id), 0.0)
            
            # Calculate node importance
            importance = self.graph.degree(node_id)
            
            # Get cluster if available
            cluster = node_data.get('cluster', -1)
            
            # Add result with enhanced metadata
            results.append({
                'id': node_id,
                'text': node_data['text'],
                'keywords': node_data['keywords'],
                'similarity': similarity,
                'importance': importance,
                'cluster': cluster
            })
        
        # Sort results by similarity score
        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results

    def visualize(self, output_path: Optional[str] = None):
        """
        Visualize the knowledge graph
        
        Args:
            output_path: Optional path to save visualization
        """
        plt.figure(figsize=(15, 10))
        
        # Use Kamada-Kawai layout for better visualization of connected components
        pos = nx.kamada_kawai_layout(self.graph)
        
        # Get cluster information if available
        clusters = [self.graph.nodes[n].get('cluster', 0) for n in self.graph.nodes()]
        unique_clusters = set(clusters)
        
        # Create color map based on clusters
        cmap = plt.cm.tab10
        node_colors = [cmap(clusters[n] % 10) for n in self.graph.nodes()]
        
        # Scale node sizes by degree centrality
        node_sizes = [3000 * (self.graph.degree(node) / max(self.graph.degree(), key=lambda x: x[1])[1])
                     for node in self.graph.nodes()]
        node_sizes = [max(500, s) for s in node_sizes]  # Minimum size
        
        # Draw nodes
        nx.draw_networkx_nodes(self.graph, pos,
                             node_color=node_colors,
                             node_size=node_sizes,
                             alpha=0.8)
        
        # Draw edges with width proportional to weight
        edges = [(u, v) for u, v in self.graph.edges()]
        edge_weights = [self.graph.get_edge_data(u, v)['weight'] * 2 for u, v in edges]
        
        nx.draw_networkx_edges(self.graph, pos,
                             edgelist=edges,
                             width=edge_weights,
                             alpha=0.5,
                             edge_color='gray',
                             arrows=True,
                             arrowsize=15)
        
        # Draw labels
        labels = {node: f"{node}" for node in self.graph.nodes()}
        nx.draw_networkx_labels(self.graph, pos,
                               labels=labels,
                               font_size=10,
                               font_weight='bold')
        
        # Add legend for clusters
        if len(unique_clusters) > 1:
            legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                        label=f'Küme {c}', 
                                        markerfacecolor=cmap(c % 10), 
                                        markersize=10) 
                             for c in sorted(unique_clusters)]
            plt.legend(handles=legend_elements, loc='upper right')
        
        plt.title("Türkçe Bilgi Grafı")
        plt.axis('off')
        
        # Save or show
        if output_path:
            plt.savefig(output_path, bbox_inches='tight')
            print(f"Grafik kaydedildi: {output_path}")
        else:
            plt.show()
    
    def export_graph(self, output_dir: str):
        """
        Export graph data to files
        
        Args:
            output_dir: Directory to save graph data
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Export graph structure
        nx.write_gexf(self.graph, output_path / "knowledge_graph.gexf")
        
        # Export node data as text
        with open(output_path / "nodes.txt", "w", encoding="utf-8") as f:
            for i in range(len(self.documents)):
                node = self.graph.nodes[i]
                f.write(f"Node {i}:\n")
                f.write(f"Text: {node['text']}\n")
                f.write(f"Keywords: {', '.join(node['keywords'])}\n")
                f.write(f"Connections: {list(self.graph.neighbors(i))}\n")
                f.write("-" * 50 + "\n")
        
        # Export graph metrics
        metrics = self.graph_analyzer.analyze_graph_structure(self.graph)
        with open(output_path / "graph_metrics.txt", "w", encoding="utf-8") as f:
            for key, value in metrics.items():
                f.write(f"{key}: {value}\n")
        
        print(f"Graf verileri dışa aktarıldı: {output_dir}")


# Kullanım örneği:
if __name__ == "__main__":
    # Test belgeleri
    documents = [
        "Yapay zeka, insan zekasını taklit eden ve öğrenebilen sistemlerdir.",
        "Derin öğrenme, yapay sinir ağları kullanarak karmaşık örüntüleri tespit eder.",
        "Doğal dil işleme, bilgisayarların Türkçe dahil insan dillerini anlamasını sağlar.",
        "Makine öğrenimi, verilerden öğrenerek kendini geliştiren algoritmaları kapsar.",
        "Bilgisayarlı görü, görüntü ve videoları yapay zeka ile analiz eder.",
        "Türkçe doğal dil işleme, Türkçe metinleri anlama ve işleme yeteneğidir.",
        "Türk dili, Ural-Altay dil ailesine mensup sondan eklemeli bir dildir.",
        "Transformatör mimarileri, doğal dil işlemede çığır açan bir tekniktir.",
        "BERT modeli, çift yönlü kodlama yaparak bağlam analizini iyileştirir.",
        "GPT modelleri, ileriye doğru tahmin yaparak tutarlı metinler üretir."
    ]
    
    # RAG sistemi oluştur (tokenizer aktif)
    rag = EnhancedTurkishGraphRAG(use_tokenizer=True)
    
    # Bilgi grafını oluştur
    print("Bilgi grafı oluşturuluyor...")
    rag.build_knowledge_graph(documents)
    
    # Örnek sorgu
    query = "Yapay zeka ve doğal dil işleme arasındaki ilişki nedir?"
    print(f"\nSorgu: {query}")
    
    # Sorguyu işle
    results = rag.query(query)
    
    # Sonuçları göster
    print("\nSonuçlar:")
    for r in results:
        print(f"\nBelge {r['id']}:")
        print(f"Metin: {r['text']}")
        print(f"Anahtar Kelimeler: {', '.join(r['keywords'])}")
        print(f"Benzerlik: {r['similarity']:.3f}")
        print(f"Önem derecesi: {r['importance']}")
        if 'cluster' in r:
            print(f"Küme: {r['cluster']}")
    
    # Grafı görselleştir
    print("\nGraf görselleştiriliyor...")
    rag.visualize()
    
    # Graf verilerini dışa aktar
    rag.export_graph("graph_export")