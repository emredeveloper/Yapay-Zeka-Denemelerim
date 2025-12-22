"""
CLaRa Evaluation - Base vs Trained Model Karşılaştırması
=========================================================
Eğitilmiş CLaRa modeli ile base RAG sistemini karşılaştırır.
"""

import torch
import torch.nn.functional as F
import pickle
import json
import os
import time
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
import ollama
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress
from rich import print as rprint

console = Console()

# ============================================================================
# Veri Yapıları
# ============================================================================

@dataclass
class EvalResult:
    """Değerlendirme sonucu"""
    question: str
    clara_answer: str
    base_answer: str
    clara_scores: List[float]
    base_scores: List[float]
    clara_time: float
    base_time: float
    clara_sources: List[str]
    base_sources: List[str]


# ============================================================================
# Base RAG Sistemi (Eğitimsiz)
# ============================================================================

class BaseRAG:
    """
    Basit RAG sistemi - CLaRa eğitimi olmadan.
    Sadece embedding benzerliği kullanır.
    """
    
    def __init__(self, documents: List[Dict], embedding_model: str = "nomic-embed-text-v2-moe:latest", 
                 llm_model: str = "granite4:3b"):
        self.documents = documents
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def _encode(self, text: str) -> torch.Tensor:
        """Metni embedding'e çevirir"""
        truncated = text[:2000] if len(text) > 2000 else text
        try:
            response = ollama.embeddings(model=self.embedding_model, prompt=truncated)
            return torch.tensor(response["embedding"], dtype=torch.float32)
        except:
            response = ollama.embeddings(model=self.embedding_model, prompt=text[:500])
            return torch.tensor(response["embedding"], dtype=torch.float32)
    
    def query(self, question: str, top_k: int = 5) -> Dict[str, Any]:
        """Sorgu yapar - sadece embedding benzerliği ile"""
        start_time = time.time()
        
        # Query embedding
        query_emb = self._encode(question)
        
        # Tüm belgelerle benzerlik hesapla
        scores = []
        for doc in self.documents:
            doc_emb = doc.get("embedding")
            if doc_emb is not None:
                if not isinstance(doc_emb, torch.Tensor):
                    doc_emb = torch.tensor(doc_emb, dtype=torch.float32)
                
                # Cosine similarity
                similarity = F.cosine_similarity(
                    query_emb.unsqueeze(0),
                    doc_emb.unsqueeze(0)
                ).item()
                scores.append((doc, similarity))
        
        # Sırala
        scores.sort(key=lambda x: x[1], reverse=True)
        top_docs = scores[:top_k]
        
        # Context oluştur
        context_parts = [doc["text"] for doc, _ in top_docs if len(doc["text"]) > 50]
        context = "\n\n---\n\n".join(context_parts[:3])
        
        # Cevap üret
        answer = self._generate_answer(question, context)
        
        elapsed = time.time() - start_time
        
        return {
            "answer": answer,
            "scores": [s for _, s in top_docs],
            "sources": [doc["text"][:150] for doc, _ in top_docs],
            "time": elapsed
        }
    
    def _generate_answer(self, question: str, context: str) -> str:
        """LLM ile cevap üret"""
        prompt = f"""Aşağıdaki bağlamı kullanarak soruyu yanıtla.

BAĞLAM:
{context[:3000]}

SORU: {question}

CEVAP:"""
        
        try:
            response = ollama.generate(
                model=self.llm_model,
                prompt=prompt,
                options={"temperature": 0.3, "num_predict": 400}
            )
            return response["response"].strip()
        except Exception as e:
            return f"Hata: {str(e)}"


# ============================================================================
# CLaRa RAG Sistemi (Eğitilmiş)
# ============================================================================

class CLaRaRAGEval:
    """CLaRa RAG sistemi - evaluation için"""
    
    def __init__(self, model_path: str, llm_model: str = "granite4:3b"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.llm_model = llm_model
        self.embedding_model = "nomic-embed-text-v2-moe:latest"
        
        # Model yükle
        self._load_model(model_path)
    
    def _load_model(self, model_path: str):
        """Kaydedilmiş modeli yükle"""
        from clara_full_implementation import CLaRaModel, TrainingConfig
        
        # Config
        config_path = os.path.join(model_path, "config.json")
        with open(config_path, "r") as f:
            config_dict = json.load(f)
        
        self.config = TrainingConfig(
            embedding_dim=config_dict.get("embedding_dim", 768),
            hidden_dim=config_dict.get("hidden_dim", 512),
            compressed_dim=config_dict.get("compressed_dim", 256),
            num_attention_heads=config_dict.get("num_attention_heads", 8),
            num_layers=config_dict.get("num_layers", 2),
            top_k=config_dict.get("top_k", 10)
        )
        
        self.embedding_model = config_dict.get("embedding_model", "nomic-embed-text-v2-moe:latest")
        self.llm_model = config_dict.get("llm_model", "granite4:3b")
        
        # Model
        self.model = CLaRaModel(self.config).to(self.device)
        
        model_file = os.path.join(model_path, "clara_model.pt")
        if os.path.exists(model_file):
            checkpoint = torch.load(model_file, map_location=self.device, weights_only=False)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
            else:
                self.model.load_state_dict(checkpoint)
        
        self.model.eval()
        
        # Belgeler
        docs_path = os.path.join(model_path, "documents.pkl")
        with open(docs_path, "rb") as f:
            self.documents = pickle.load(f)
    
    def _encode(self, text: str) -> torch.Tensor:
        """Metni embedding'e çevirir"""
        truncated = text[:2000] if len(text) > 2000 else text
        try:
            response = ollama.embeddings(model=self.embedding_model, prompt=truncated)
            return torch.tensor(response["embedding"], dtype=torch.float32)
        except:
            response = ollama.embeddings(model=self.embedding_model, prompt=text[:500])
            return torch.tensor(response["embedding"], dtype=torch.float32)
    
    def query(self, question: str, top_k: int = 5) -> Dict[str, Any]:
        """CLaRa ile sorgu yapar"""
        start_time = time.time()
        
        with torch.no_grad():
            # Query embedding ve sıkıştırma
            query_emb = self._encode(question).to(self.device)
            query_compressed, _ = self.model.compressor(query_emb.unsqueeze(0))
            query_compressed = query_compressed.squeeze(0)
            
            # Belge skorları - model ile sıkıştır
            scores = []
            for doc in self.documents:
                orig_emb = doc.get("embedding")
                if orig_emb is not None:
                    if not isinstance(orig_emb, torch.Tensor):
                        orig_emb = torch.tensor(orig_emb, dtype=torch.float32)
                    orig_emb = orig_emb.to(self.device)
                    
                    # Model ile sıkıştır
                    if orig_emb.dim() == 1:
                        orig_emb = orig_emb.unsqueeze(0)
                    comp_emb, _ = self.model.compressor(orig_emb)
                    comp_emb = comp_emb.squeeze(0)
                    
                    # Cosine similarity
                    similarity = F.cosine_similarity(
                        query_compressed.unsqueeze(0),
                        comp_emb.unsqueeze(0)
                    ).item()
                    scores.append((doc, similarity))
            
            # Sırala
            scores.sort(key=lambda x: x[1], reverse=True)
            top_docs = scores[:top_k]
        
        # Context oluştur
        context_parts = [doc["text"] for doc, _ in top_docs if len(doc["text"]) > 50]
        context = "\n\n---\n\n".join(context_parts[:3])
        
        # Cevap üret
        answer = self._generate_answer(question, context)
        
        elapsed = time.time() - start_time
        
        return {
            "answer": answer,
            "scores": [s for _, s in top_docs],
            "sources": [doc["text"][:150] for doc, _ in top_docs],
            "time": elapsed
        }
    
    def _generate_answer(self, question: str, context: str) -> str:
        """LLM ile cevap üret"""
        prompt = f"""Sen bir araştırma asistanısın. Aşağıdaki bağlam bilgilerini kullanarak soruyu yanıtla.

KURALLAR:
- SADECE bağlamdaki bilgileri kullan
- Makale başlığı sorulursa tam başlığı yaz
- Türkçe açıklama yap

BAĞLAM:
{context[:4000]}

SORU: {question}

CEVAP:"""
        
        try:
            response = ollama.generate(
                model=self.llm_model,
                prompt=prompt,
                options={"temperature": 0.3, "num_predict": 400}
            )
            return response["response"].strip()
        except Exception as e:
            return f"Hata: {str(e)}"


# ============================================================================
# Evaluation Metrikleri
# ============================================================================

def calculate_metrics(clara_results: List[Dict], base_results: List[Dict]) -> Dict:
    """Karşılaştırma metrikleri hesapla"""
    
    # Ortalama skorlar
    clara_avg_score = sum(r["scores"][0] for r in clara_results) / len(clara_results)
    base_avg_score = sum(r["scores"][0] for r in base_results) / len(base_results)
    
    # Ortalama süreler
    clara_avg_time = sum(r["time"] for r in clara_results) / len(clara_results)
    base_avg_time = sum(r["time"] for r in base_results) / len(base_results)
    
    # Skor varyansı (tutarlılık)
    clara_score_var = sum((r["scores"][0] - clara_avg_score)**2 for r in clara_results) / len(clara_results)
    base_score_var = sum((r["scores"][0] - base_avg_score)**2 for r in base_results) / len(base_results)
    
    # Top-1 vs Top-5 skor farkı (ayrım gücü)
    clara_discrimination = sum(r["scores"][0] - r["scores"][-1] for r in clara_results) / len(clara_results)
    base_discrimination = sum(r["scores"][0] - r["scores"][-1] for r in base_results) / len(base_results)
    
    return {
        "clara_avg_score": clara_avg_score,
        "base_avg_score": base_avg_score,
        "clara_avg_time": clara_avg_time,
        "base_avg_time": base_avg_time,
        "clara_score_variance": clara_score_var,
        "base_score_variance": base_score_var,
        "clara_discrimination": clara_discrimination,
        "base_discrimination": base_discrimination
    }


# ============================================================================
# Ana Evaluation Fonksiyonu
# ============================================================================

def run_evaluation(model_path: str = "clara_saved_model"):
    """Tam evaluation çalıştır"""
    
    console.print(Panel.fit(
        "[bold cyan]CLaRa Evaluation[/bold cyan]\n"
        "[dim]Base RAG vs Trained CLaRa Karşılaştırması[/dim]",
        border_style="cyan"
    ))
    
    # Model kontrol
    if not os.path.exists(os.path.join(model_path, "clara_model.pt")):
        console.print("[red]❌ Eğitilmiş model bulunamadı![/red]")
        console.print(f"[dim]Beklenen konum: {model_path}[/dim]")
        return
    
    # Sistemleri yükle
    console.print("\n[yellow]📦 Sistemler yükleniyor...[/yellow]")
    
    # CLaRa yükle
    console.print("   [dim]CLaRa model yükleniyor...[/dim]")
    clara = CLaRaRAGEval(model_path)
    
    # Base RAG oluştur (aynı belgelerle)
    console.print("   [dim]Base RAG hazırlanıyor...[/dim]")
    base = BaseRAG(clara.documents, clara.embedding_model, clara.llm_model)
    
    console.print(f"[green]✅ {len(clara.documents)} belge yüklendi[/green]")
    
    # Test soruları
    test_questions = [
        "Bu makaleler ne hakkında?",
        "Makale başlıkları nelerdir?",
        "Hangi modeller kullanılmış?",
        "Temel katkılar nelerdir?",
        "Deneysel sonuçlar nasıl?",
        "Gemma Scope 2 nedir?",
        "SAE (Sparse Autoencoder) ne işe yarar?",
        "Transformer mimarisi nasıl kullanılmış?"
    ]
    
    # Evaluation
    console.print("\n[yellow]🔍 Evaluation başlıyor...[/yellow]\n")
    
    clara_results = []
    base_results = []
    
    with Progress() as progress:
        task = progress.add_task("[cyan]Sorgular işleniyor...", total=len(test_questions))
        
        for question in test_questions:
            # CLaRa sorgu
            clara_result = clara.query(question, top_k=5)
            clara_results.append(clara_result)
            
            # Base sorgu
            base_result = base.query(question, top_k=5)
            base_results.append(base_result)
            
            progress.update(task, advance=1)
    
    # Metrikleri hesapla
    metrics = calculate_metrics(clara_results, base_results)
    
    # Sonuçları göster
    console.print("\n" + "="*70)
    console.print("[bold cyan]📊 EVALUATION SONUÇLARI[/bold cyan]")
    console.print("="*70)
    
    # Metrik tablosu
    metric_table = Table(title="Karşılaştırma Metrikleri", show_header=True)
    metric_table.add_column("Metrik", style="cyan")
    metric_table.add_column("CLaRa", style="green")
    metric_table.add_column("Base", style="yellow")
    metric_table.add_column("Fark", style="magenta")
    
    metric_table.add_row(
        "Ortalama Top-1 Skor",
        f"{metrics['clara_avg_score']:.4f}",
        f"{metrics['base_avg_score']:.4f}",
        f"{metrics['clara_avg_score'] - metrics['base_avg_score']:+.4f}"
    )
    metric_table.add_row(
        "Ortalama Süre (sn)",
        f"{metrics['clara_avg_time']:.2f}",
        f"{metrics['base_avg_time']:.2f}",
        f"{metrics['clara_avg_time'] - metrics['base_avg_time']:+.2f}"
    )
    metric_table.add_row(
        "Skor Varyansı",
        f"{metrics['clara_score_variance']:.4f}",
        f"{metrics['base_score_variance']:.4f}",
        f"{metrics['clara_score_variance'] - metrics['base_score_variance']:+.4f}"
    )
    metric_table.add_row(
        "Ayrım Gücü (Top1-Top5)",
        f"{metrics['clara_discrimination']:.4f}",
        f"{metrics['base_discrimination']:.4f}",
        f"{metrics['clara_discrimination'] - metrics['base_discrimination']:+.4f}"
    )
    
    console.print(metric_table)
    
    # Detaylı sonuçlar
    console.print("\n" + "="*70)
    console.print("[bold cyan]📝 DETAYLI SORGU SONUÇLARI[/bold cyan]")
    console.print("="*70)
    
    for i, question in enumerate(test_questions):
        clara_r = clara_results[i]
        base_r = base_results[i]
        
        console.print(f"\n[bold]━━━ Soru {i+1}: {question} ━━━[/bold]")
        
        # Skorlar
        score_table = Table(show_header=True, box=None)
        score_table.add_column("", width=10)
        score_table.add_column("Top-1", width=10)
        score_table.add_column("Top-5", width=10)
        score_table.add_column("Süre", width=10)
        
        score_table.add_row(
            "[green]CLaRa[/green]",
            f"{clara_r['scores'][0]:.4f}",
            f"{clara_r['scores'][-1]:.4f}",
            f"{clara_r['time']:.2f}s"
        )
        score_table.add_row(
            "[yellow]Base[/yellow]",
            f"{base_r['scores'][0]:.4f}",
            f"{base_r['scores'][-1]:.4f}",
            f"{base_r['time']:.2f}s"
        )
        
        console.print(score_table)
        
        # Cevaplar (kısa)
        console.print(f"\n[green]CLaRa Cevap:[/green] {clara_r['answer'][:200]}...")
        console.print(f"[yellow]Base Cevap:[/yellow] {base_r['answer'][:200]}...")
    
    # Özet
    console.print("\n" + "="*70)
    console.print("[bold cyan]📈 ÖZET[/bold cyan]")
    console.print("="*70)
    
    score_diff = metrics['clara_avg_score'] - metrics['base_avg_score']
    disc_diff = metrics['clara_discrimination'] - metrics['base_discrimination']
    
    if score_diff > 0:
        console.print(f"[green]✅ CLaRa ortalama skorda %{abs(score_diff)*100:.1f} daha iyi[/green]")
    else:
        console.print(f"[yellow]⚠️ Base model ortalama skorda %{abs(score_diff)*100:.1f} daha iyi[/yellow]")
    
    if disc_diff > 0:
        console.print(f"[green]✅ CLaRa belge ayrımında %{abs(disc_diff)*100:.1f} daha iyi[/green]")
    else:
        console.print(f"[yellow]⚠️ Base model belge ayrımında %{abs(disc_diff)*100:.1f} daha iyi[/yellow]")
    
    # Sonuçları kaydet
    save_results(test_questions, clara_results, base_results, metrics)
    
    return metrics


def save_results(questions: List[str], clara_results: List[Dict], 
                 base_results: List[Dict], metrics: Dict):
    """Sonuçları JSON'a kaydet"""
    
    results = {
        "metrics": metrics,
        "queries": []
    }
    
    for i, q in enumerate(questions):
        results["queries"].append({
            "question": q,
            "clara": {
                "answer": clara_results[i]["answer"],
                "top_score": clara_results[i]["scores"][0],
                "time": clara_results[i]["time"]
            },
            "base": {
                "answer": base_results[i]["answer"],
                "top_score": base_results[i]["scores"][0],
                "time": base_results[i]["time"]
            }
        })
    
    with open("clara_evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    console.print(f"\n[dim]📁 Sonuçlar kaydedildi: clara_evaluation_results.json[/dim]")


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    import sys
    
    model_path = sys.argv[1] if len(sys.argv) > 1 else "clara_saved_model"
    
    run_evaluation(model_path)

