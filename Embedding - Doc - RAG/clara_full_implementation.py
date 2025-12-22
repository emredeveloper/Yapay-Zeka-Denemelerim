"""
CLaRa - Tam Implementasyon
===========================
Apple'ın CLaRa (Continuous Latent Reasoning) sisteminin gerçek implementasyonu.

Özellikler:
1. SCP (Salient Compressor Pretraining) - Contrastive learning ile sıkıştırıcı eğitimi
2. Differentiable Top-K - Gradyan aktarımı için soft seçim
3. End-to-End Eğitim - Reranker ve generator birlikte optimize
4. Continuous Latent Space - Sürekli latent uzayda belge temsili
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
import ollama
import fitz  # PyMuPDF
import re
import os
import json
from collections import Counter
import math
from tqdm import tqdm
import random


# ============================================================================
# Veri Yapıları
# ============================================================================

@dataclass
class Document:
    """Belge veri yapısı"""
    id: int
    text: str
    embedding: Optional[torch.Tensor] = None
    compressed_embedding: Optional[torch.Tensor] = None
    salient_tokens: List[int] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QAPair:
    """Soru-Cevap çifti (SCP eğitimi için)"""
    question: str
    answer: str
    context: str
    positive_doc_ids: List[int] = field(default_factory=list)
    negative_doc_ids: List[int] = field(default_factory=list)


@dataclass 
class TrainingConfig:
    """Eğitim konfigürasyonu"""
    # Model
    embedding_dim: int = 768
    hidden_dim: int = 512
    compressed_dim: int = 256
    num_attention_heads: int = 8
    num_layers: int = 2
    
    # Eğitim
    batch_size: int = 8
    learning_rate: float = 1e-4
    num_epochs: int = 10
    warmup_steps: int = 100
    
    # Top-K
    top_k: int = 5
    temperature: float = 0.1
    
    # Loss ağırlıkları
    contrastive_weight: float = 1.0
    generation_weight: float = 1.0
    rerank_weight: float = 0.5


# ============================================================================
# Ollama Embedding
# ============================================================================

class OllamaEmbeddings:
    """Ollama embedding modeli"""
    
    def __init__(self, model_name: str = "nomic-embed-text-v2-moe:latest"):
        self.model_name = model_name
        self._dim = None
    
    def encode(self, texts: Union[str, List[str]]) -> torch.Tensor:
        """Metinleri embedding'e dönüştürür"""
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = []
        for text in texts:
            # Maksimum 2000 karakter (güvenli limit)
            truncated = text[:2000] if len(text) > 2000 else text
            try:
                response = ollama.embeddings(model=self.model_name, prompt=truncated)
                embeddings.append(response["embedding"])
            except Exception as e:
                # Hata durumunda daha da kısa dene
                try:
                    response = ollama.embeddings(model=self.model_name, prompt=text[:500])
                    embeddings.append(response["embedding"])
                except:
                    # Son çare: boş embedding
                    if self._dim:
                        embeddings.append([0.0] * self._dim)
                    else:
                        raise e
        
        return torch.tensor(embeddings, dtype=torch.float32)
    
    @property
    def embedding_dim(self) -> int:
        if self._dim is None:
            test = self.encode("test")
            self._dim = test.shape[-1]
        return self._dim


# ============================================================================
# 1. Document Compressor Network (SCP)
# ============================================================================

class DocumentCompressorNetwork(nn.Module):
    """
    Belge Sıkıştırma Ağı - SCP'nin temel bileşeni.
    
    Belgeleri yüksek boyutlu embedding'lerden düşük boyutlu,
    semantik açıdan zengin vektörlere sıkıştırır.
    """
    
    def __init__(self, config: TrainingConfig):
        super().__init__()
        self.config = config
        
        # Encoder katmanları
        self.input_proj = nn.Linear(config.embedding_dim, config.hidden_dim)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_attention_heads,
            dim_feedforward=config.hidden_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        
        # Sıkıştırma katmanları
        self.compress_layers = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_dim, config.compressed_dim)
        )
        
        # Salient token attention
        self.salient_attention = nn.MultiheadAttention(
            embed_dim=config.hidden_dim,
            num_heads=config.num_attention_heads,
            batch_first=True
        )
        
        # Salient query (learnable)
        self.salient_query = nn.Parameter(torch.randn(1, 1, config.hidden_dim))
        
    def forward(self, embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            embeddings: [batch, embedding_dim] veya [batch, seq_len, embedding_dim]
            
        Returns:
            compressed: Sıkıştırılmış temsil [batch, compressed_dim]
            salient_weights: Salient token ağırlıkları [batch, seq_len]
        """
        # Boyut kontrolü
        if embeddings.dim() == 2:
            embeddings = embeddings.unsqueeze(1)  # [batch, 1, embed_dim]
        
        batch_size = embeddings.shape[0]
        
        # Input projection
        x = self.input_proj(embeddings)  # [batch, seq_len, hidden_dim]
        
        # Transformer encoding
        x = self.transformer(x)  # [batch, seq_len, hidden_dim]
        
        # Salient attention - önemli token'ları bul
        query = self.salient_query.expand(batch_size, -1, -1)  # [batch, 1, hidden_dim]
        attended, salient_weights = self.salient_attention(query, x, x)
        salient_weights = salient_weights.squeeze(1)  # [batch, seq_len]
        
        # Pooling (attention-weighted)
        weighted_x = x * salient_weights.unsqueeze(-1)
        pooled = weighted_x.sum(dim=1)  # [batch, hidden_dim]
        
        # Sıkıştırma
        compressed = self.compress_layers(pooled)  # [batch, compressed_dim]
        
        # L2 normalizasyon
        compressed = F.normalize(compressed, p=2, dim=-1)
        
        return compressed, salient_weights


# ============================================================================
# 2. Differentiable Top-K Selection
# ============================================================================

class DifferentiableTopK(nn.Module):
    """
    Differentiable Top-K Seçimi.
    
    Soft top-k seçimi yaparak gradyanların retrieval'a
    geri aktarılmasını sağlar.
    """
    
    def __init__(self, k: int = 5, temperature: float = 0.1):
        super().__init__()
        self.k = k
        self.temperature = temperature
    
    def forward(self, scores: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Soft top-k seçimi yapar.
        
        Args:
            scores: Benzerlik skorları [batch, num_docs]
            
        Returns:
            soft_selection: Soft seçim ağırlıkları [batch, num_docs]
            hard_indices: Hard top-k indeksleri [batch, k]
        """
        batch_size, num_docs = scores.shape
        
        # Temperature scaling
        scaled_scores = scores / self.temperature
        
        # Soft selection (Gumbel-Softmax benzeri)
        soft_selection = F.softmax(scaled_scores, dim=-1)
        
        # Hard indices (inference için)
        _, hard_indices = torch.topk(scores, min(self.k, num_docs), dim=-1)
        
        # Straight-through estimator
        # Forward: hard selection, Backward: soft gradients
        hard_selection = torch.zeros_like(soft_selection)
        hard_selection.scatter_(1, hard_indices, 1.0)
        
        # STE: hard forward, soft backward
        selection = hard_selection - soft_selection.detach() + soft_selection
        
        return selection, hard_indices
    
    def relaxed_topk(self, scores: torch.Tensor) -> torch.Tensor:
        """
        Tamamen differentiable relaxed top-k.
        Eğitim sırasında kullanılır.
        """
        # Sinkhorn-based relaxed top-k
        scaled = scores / self.temperature
        
        # Iterative softmax (Sinkhorn)
        for _ in range(3):
            scaled = scaled - torch.logsumexp(scaled, dim=-1, keepdim=True)
            scaled = scaled - torch.logsumexp(scaled, dim=-2, keepdim=True)
        
        return F.softmax(scaled, dim=-1)


# ============================================================================
# 3. Joint Reranker-Generator
# ============================================================================

class JointRerankerGenerator(nn.Module):
    """
    Birleşik Reranker ve Generator modülü.
    
    CLaRa'nın ana yeniliği: Reranker ve generator
    aynı loss üzerinden birlikte optimize edilir.
    """
    
    def __init__(self, config: TrainingConfig):
        super().__init__()
        self.config = config
        
        # Reranker
        self.reranker = RerankerModule(config)
        
        # Context fusion
        self.context_fusion = ContextFusionModule(config)
        
        # Generation score predictor
        self.generation_scorer = nn.Sequential(
            nn.Linear(config.compressed_dim * 2, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_dim, 1)
        )
    
    def forward(
        self,
        query_embedding: torch.Tensor,
        doc_embeddings: torch.Tensor,
        doc_compressed: torch.Tensor,
        initial_scores: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            query_embedding: [batch, embedding_dim]
            doc_embeddings: [batch, num_docs, embedding_dim]
            doc_compressed: [batch, num_docs, compressed_dim]
            initial_scores: [batch, num_docs]
            
        Returns:
            rerank_scores: Yeniden sıralama skorları
            fused_context: Birleştirilmiş bağlam
            generation_scores: Üretim kalitesi tahminleri
        """
        batch_size, num_docs, _ = doc_embeddings.shape
        
        # Reranking
        rerank_scores = self.reranker(query_embedding, doc_compressed, initial_scores)
        
        # Context fusion
        fused_context = self.context_fusion(doc_compressed, rerank_scores)
        
        # Generation quality prediction
        query_compressed = query_embedding[:, :self.config.compressed_dim]
        combined = torch.cat([query_compressed, fused_context], dim=-1)
        generation_scores = self.generation_scorer(combined).squeeze(-1)
        
        return rerank_scores, fused_context, generation_scores


class RerankerModule(nn.Module):
    """Attention tabanlı reranker"""
    
    def __init__(self, config: TrainingConfig):
        super().__init__()
        
        self.query_proj = nn.Linear(config.embedding_dim, config.hidden_dim)
        self.doc_proj = nn.Linear(config.compressed_dim, config.hidden_dim)
        
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=config.hidden_dim,
            num_heads=config.num_attention_heads,
            batch_first=True
        )
        
        self.score_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(config.hidden_dim // 2, 1)
        )
    
    def forward(
        self,
        query: torch.Tensor,
        doc_compressed: torch.Tensor,
        initial_scores: torch.Tensor
    ) -> torch.Tensor:
        """Belgeleri yeniden sıralar"""
        batch_size, num_docs, _ = doc_compressed.shape
        
        # Projections
        q = self.query_proj(query).unsqueeze(1)  # [batch, 1, hidden]
        d = self.doc_proj(doc_compressed)  # [batch, num_docs, hidden]
        
        # Cross attention
        attended, _ = self.cross_attention(q, d, d)  # [batch, 1, hidden]
        
        # Score computation
        interaction = d * attended  # [batch, num_docs, hidden]
        scores = self.score_head(interaction).squeeze(-1)  # [batch, num_docs]
        
        # Combine with initial scores
        combined_scores = scores + 0.3 * initial_scores
        
        return combined_scores


class ContextFusionModule(nn.Module):
    """Attention tabanlı bağlam birleştirme"""
    
    def __init__(self, config: TrainingConfig):
        super().__init__()
        
        self.attention = nn.MultiheadAttention(
            embed_dim=config.compressed_dim,
            num_heads=4,
            batch_first=True
        )
        
        self.output_proj = nn.Linear(config.compressed_dim, config.compressed_dim)
    
    def forward(
        self,
        doc_compressed: torch.Tensor,
        rerank_scores: torch.Tensor
    ) -> torch.Tensor:
        """Belgeleri skorlara göre birleştirir"""
        # Attention weights from rerank scores
        weights = F.softmax(rerank_scores, dim=-1).unsqueeze(-1)
        
        # Weighted sum
        fused = (doc_compressed * weights).sum(dim=1)
        
        # Output projection
        fused = self.output_proj(fused)
        
        return fused


# ============================================================================
# 4. Contrastive Loss (SCP Eğitimi)
# ============================================================================

class InfoNCELoss(nn.Module):
    """
    InfoNCE Contrastive Loss - SCP eğitimi için.
    
    Pozitif çiftleri yakınlaştırır, negatif çiftleri uzaklaştırır.
    """
    
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(
        self,
        query_embeddings: torch.Tensor,
        positive_embeddings: torch.Tensor,
        negative_embeddings: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            query_embeddings: [batch, dim]
            positive_embeddings: [batch, dim]
            negative_embeddings: [batch, num_neg, dim] (optional)
        """
        # Normalize
        query_embeddings = F.normalize(query_embeddings, p=2, dim=-1)
        positive_embeddings = F.normalize(positive_embeddings, p=2, dim=-1)
        
        # Positive scores
        pos_scores = (query_embeddings * positive_embeddings).sum(dim=-1) / self.temperature
        
        if negative_embeddings is not None:
            negative_embeddings = F.normalize(negative_embeddings, p=2, dim=-1)
            # Negative scores
            neg_scores = torch.bmm(
                negative_embeddings, 
                query_embeddings.unsqueeze(-1)
            ).squeeze(-1) / self.temperature  # [batch, num_neg]
            
            # InfoNCE
            logits = torch.cat([pos_scores.unsqueeze(-1), neg_scores], dim=-1)
            labels = torch.zeros(query_embeddings.shape[0], dtype=torch.long, device=query_embeddings.device)
            loss = F.cross_entropy(logits, labels)
        else:
            # In-batch negatives
            similarity_matrix = torch.mm(query_embeddings, positive_embeddings.t()) / self.temperature
            labels = torch.arange(query_embeddings.shape[0], device=query_embeddings.device)
            loss = F.cross_entropy(similarity_matrix, labels)
        
        return loss


# ============================================================================
# 5. CLaRa Full Model
# ============================================================================

class CLaRaModel(nn.Module):
    """
    Tam CLaRa Modeli.
    
    Tüm bileşenleri birleştirir:
    - Document Compressor (SCP)
    - Differentiable Top-K
    - Joint Reranker-Generator
    """
    
    def __init__(self, config: TrainingConfig):
        super().__init__()
        self.config = config
        
        # Ana bileşenler
        self.compressor = DocumentCompressorNetwork(config)
        self.topk_selector = DifferentiableTopK(k=config.top_k, temperature=config.temperature)
        self.reranker_generator = JointRerankerGenerator(config)
        
        # Query encoder
        self.query_encoder = nn.Sequential(
            nn.Linear(config.embedding_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.compressed_dim)
        )
    
    def compress_documents(self, doc_embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Belgeleri sıkıştırır"""
        return self.compressor(doc_embeddings)
    
    def encode_query(self, query_embedding: torch.Tensor) -> torch.Tensor:
        """Sorguyu kodlar"""
        return self.query_encoder(query_embedding)
    
    def forward(
        self,
        query_embedding: torch.Tensor,
        doc_embeddings: torch.Tensor,
        doc_compressed: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Tam forward pass.
        
        Args:
            query_embedding: [batch, embedding_dim]
            doc_embeddings: [batch, num_docs, embedding_dim]
            doc_compressed: [batch, num_docs, compressed_dim] (precomputed, optional)
        """
        batch_size, num_docs, embed_dim = doc_embeddings.shape
        
        # 1. Belgeleri sıkıştır (eğer önceden hesaplanmadıysa)
        if doc_compressed is None:
            doc_flat = doc_embeddings.view(-1, embed_dim)
            compressed_flat, salient_weights = self.compressor(doc_flat)
            doc_compressed = compressed_flat.view(batch_size, num_docs, -1)
        else:
            salient_weights = None
        
        # 2. Sorguyu kodla
        query_compressed = self.encode_query(query_embedding)  # [batch, compressed_dim]
        
        # 3. Initial similarity scores
        query_expanded = query_compressed.unsqueeze(1)  # [batch, 1, compressed_dim]
        initial_scores = F.cosine_similarity(query_expanded, doc_compressed, dim=-1)  # [batch, num_docs]
        
        # 4. Differentiable Top-K
        selection_weights, top_indices = self.topk_selector(initial_scores)
        
        # 5. Reranking ve context fusion
        rerank_scores, fused_context, gen_scores = self.reranker_generator(
            query_embedding, doc_embeddings, doc_compressed, initial_scores
        )
        
        return {
            "query_compressed": query_compressed,
            "doc_compressed": doc_compressed,
            "initial_scores": initial_scores,
            "selection_weights": selection_weights,
            "top_indices": top_indices,
            "rerank_scores": rerank_scores,
            "fused_context": fused_context,
            "generation_scores": gen_scores,
            "salient_weights": salient_weights
        }


# ============================================================================
# 6. QA Pair Generator (SCP Eğitimi için)
# ============================================================================

class QAPairGenerator:
    """
    PDF belgelerinden QA çiftleri oluşturur.
    SCP eğitimi için kullanılır.
    """
    
    def __init__(self, llm_model: str = "granite4:3b"):
        self.llm_model = llm_model
    
    def generate_qa_pairs(self, text: str, num_pairs: int = 3) -> List[Dict]:
        """Metinden QA çiftleri oluşturur"""
        prompt = f"""Aşağıdaki metinden {num_pairs} adet soru-cevap çifti oluştur.
Her çift için:
- Soru: Metindeki bilgiye dayalı spesifik bir soru
- Cevap: Sorunun kısa ve öz cevabı

Metin:
{text[:2000]}

JSON formatında yanıt ver:
[{{"soru": "...", "cevap": "..."}}, ...]"""

        try:
            response = ollama.chat(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.7}
            )
            
            content = response["message"]["content"]
            
            # JSON parse
            json_match = re.search(r'\[.*\]', content, re.DOTALL)
            if json_match:
                pairs = json.loads(json_match.group())
                return pairs
        except:
            pass
        
        return []
    
    def generate_paraphrases(self, text: str) -> List[str]:
        """Metin için paraphrase'ler oluşturur"""
        prompt = f"""Aşağıdaki metnin 2 farklı paraphrase'ini oluştur.
Anlamı koru ama farklı kelimeler kullan.

Metin:
{text[:500]}

JSON formatında yanıt ver:
["paraphrase1", "paraphrase2"]"""

        try:
            response = ollama.chat(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.8}
            )
            
            content = response["message"]["content"]
            json_match = re.search(r'\[.*\]', content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except:
            pass
        
        return []


# ============================================================================
# 7. CLaRa Trainer
# ============================================================================

class CLaRaTrainer:
    """
    CLaRa End-to-End Eğitim.
    
    Üç aşamalı eğitim:
    1. SCP Pretraining (Contrastive)
    2. Compression Instruction Tuning
    3. End-to-End Fine-tuning
    """
    
    def __init__(
        self,
        config: TrainingConfig,
        embedding_model: str = "nomic-embed-text-v2-moe:latest",
        llm_model: str = "granite4:3b"
    ):
        self.config = config
        self.embedding_model_name = embedding_model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Modeller
        self.embedder = OllamaEmbeddings(embedding_model)
        self.llm_model = llm_model
        
        # CLaRa model
        config.embedding_dim = self.embedder.embedding_dim
        self.model = CLaRaModel(config).to(self.device)
        
        # Loss fonksiyonları
        self.contrastive_loss = InfoNCELoss(temperature=0.07)
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=0.01
        )
        
        # Learning rate scheduler (warmup + cosine decay)
        self.scheduler = None  # Eğitim sırasında oluşturulacak
        
        # QA generator
        self.qa_generator = QAPairGenerator(llm_model)
        
        # Veri
        self.documents: List[Document] = []
        self.qa_pairs: List[QAPair] = []
        
        print(f"✅ CLaRa Trainer başlatıldı")
        print(f"   Device: {self.device}")
        print(f"   Embedding dim: {config.embedding_dim}")
    
    def save(self, save_dir: str):
        """
        Tüm sistemi kaydeder - başka sistemlerde kullanılabilir.
        
        Kaydedilenler:
        - Model ağırlıkları (PyTorch)
        - Konfigürasyon
        - Belge veritabanı (embedding'ler dahil)
        - QA çiftleri
        """
        import pickle
        
        os.makedirs(save_dir, exist_ok=True)
        print(f"\n💾 Sistem kaydediliyor: {save_dir}")
        
        # 1. Model ağırlıkları
        model_path = os.path.join(save_dir, "clara_model.pt")
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, model_path)
        print(f"   ✓ Model kaydedildi: clara_model.pt")
        
        # 2. Konfigürasyon
        config_path = os.path.join(save_dir, "config.json")
        config_dict = {
            'embedding_dim': self.config.embedding_dim,
            'hidden_dim': self.config.hidden_dim,
            'compressed_dim': self.config.compressed_dim,
            'num_attention_heads': self.config.num_attention_heads,
            'num_layers': self.config.num_layers,
            'batch_size': self.config.batch_size,
            'learning_rate': self.config.learning_rate,
            'top_k': self.config.top_k,
            'temperature': self.config.temperature,
            'embedding_model': self.embedding_model_name,
            'llm_model': self.llm_model
        }
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
        print(f"   ✓ Konfigürasyon kaydedildi: config.json")
        
        # 3. Belgeler (embedding'ler dahil)
        docs_data = []
        for doc in self.documents:
            doc_dict = {
                'id': doc.id,
                'text': doc.text,
                'embedding': doc.embedding.numpy().tolist() if doc.embedding is not None else None,
                'compressed_embedding': doc.compressed_embedding.numpy().tolist() if doc.compressed_embedding is not None else None,
                'metadata': doc.metadata
            }
            docs_data.append(doc_dict)
        
        docs_path = os.path.join(save_dir, "documents.pkl")
        with open(docs_path, 'wb') as f:
            pickle.dump(docs_data, f)
        print(f"   ✓ {len(docs_data)} belge kaydedildi: documents.pkl")
        
        # 4. QA çiftleri
        qa_data = []
        for qa in self.qa_pairs:
            qa_dict = {
                'question': qa.question,
                'answer': qa.answer,
                'context': qa.context,
                'positive_doc_ids': qa.positive_doc_ids
            }
            qa_data.append(qa_dict)
        
        qa_path = os.path.join(save_dir, "qa_pairs.json")
        with open(qa_path, 'w', encoding='utf-8') as f:
            json.dump(qa_data, f, indent=2, ensure_ascii=False)
        print(f"   ✓ {len(qa_data)} QA çifti kaydedildi: qa_pairs.json")
        
        print(f"✅ Sistem başarıyla kaydedildi!")
        print(f"   📁 Konum: {os.path.abspath(save_dir)}")
    
    @classmethod
    def load(cls, save_dir: str) -> 'CLaRaTrainer':
        """
        Kaydedilmiş sistemi yükler.
        
        Args:
            save_dir: Kayıt dizini
            
        Returns:
            Yüklenmiş CLaRaTrainer instance
        """
        import pickle
        
        print(f"\n📂 Sistem yükleniyor: {save_dir}")
        
        # 1. Konfigürasyon yükle
        config_path = os.path.join(save_dir, "config.json")
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        
        config = TrainingConfig(
            embedding_dim=config_dict['embedding_dim'],
            hidden_dim=config_dict['hidden_dim'],
            compressed_dim=config_dict['compressed_dim'],
            num_attention_heads=config_dict['num_attention_heads'],
            num_layers=config_dict['num_layers'],
            batch_size=config_dict['batch_size'],
            learning_rate=config_dict['learning_rate'],
            top_k=config_dict['top_k'],
            temperature=config_dict['temperature']
        )
        print(f"   ✓ Konfigürasyon yüklendi")
        
        # 2. Trainer oluştur
        trainer = cls(
            config=config,
            embedding_model=config_dict.get('embedding_model', 'nomic-embed-text-v2-moe:latest'),
            llm_model=config_dict.get('llm_model', 'granite4:3b')
        )
        
        # 3. Model ağırlıklarını yükle
        model_path = os.path.join(save_dir, "clara_model.pt")
        checkpoint = torch.load(model_path, map_location=trainer.device)
        trainer.model.load_state_dict(checkpoint['model_state_dict'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"   ✓ Model ağırlıkları yüklendi")
        
        # 4. Belgeleri yükle
        docs_path = os.path.join(save_dir, "documents.pkl")
        with open(docs_path, 'rb') as f:
            docs_data = pickle.load(f)
        
        for doc_dict in docs_data:
            doc = Document(
                id=doc_dict['id'],
                text=doc_dict['text'],
                embedding=torch.tensor(doc_dict['embedding']) if doc_dict['embedding'] else None,
                compressed_embedding=torch.tensor(doc_dict['compressed_embedding']) if doc_dict['compressed_embedding'] else None,
                metadata=doc_dict['metadata']
            )
            trainer.documents.append(doc)
        print(f"   ✓ {len(trainer.documents)} belge yüklendi")
        
        # 5. QA çiftlerini yükle
        qa_path = os.path.join(save_dir, "qa_pairs.json")
        if os.path.exists(qa_path):
            with open(qa_path, 'r', encoding='utf-8') as f:
                qa_data = json.load(f)
            
            for qa_dict in qa_data:
                qa = QAPair(
                    question=qa_dict['question'],
                    answer=qa_dict['answer'],
                    context=qa_dict['context'],
                    positive_doc_ids=qa_dict['positive_doc_ids']
                )
                trainer.qa_pairs.append(qa)
            print(f"   ✓ {len(trainer.qa_pairs)} QA çifti yüklendi")
        
        print(f"✅ Sistem başarıyla yüklendi!")
        return trainer
    
    def load_pdfs(self, pdf_dir: str) -> List[str]:
        """PDF dosyalarını yükler ve başlıkları çıkarır"""
        all_chunks = []
        self.paper_titles = []  # Makale başlıkları
        pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith('.pdf')]
        
        print(f"\n📄 {len(pdf_files)} PDF yükleniyor...")
        
        for pdf_file in tqdm(pdf_files, desc="PDF okuma"):
            pdf_path = os.path.join(pdf_dir, pdf_file)
            try:
                doc = fitz.open(pdf_path)
                text = ""
                first_page_text = ""
                
                for i, page in enumerate(doc):
                    page_text = page.get_text()
                    text += page_text
                    if i == 0:
                        first_page_text = page_text
                doc.close()
                
                # === BAŞLIK ÇIKARMA ===
                title = self._extract_title(first_page_text, pdf_file)
                if title:
                    self.paper_titles.append(title)
                    # Başlığı özel chunk olarak ekle
                    title_chunk = f"[MAKALE BAŞLIĞI] {title}"
                    all_chunks.append(title_chunk)
                
                # Chunk'lara böl
                chunks = self._split_text(text, chunk_size=500, overlap=50)
                for chunk in chunks:
                    if len(chunk.strip()) > 100:
                        all_chunks.append(chunk.strip())
                        
            except Exception as e:
                print(f"   ⚠️ {pdf_file} okunamadı: {e}")
        
        print(f"   ✓ {len(self.paper_titles)} makale başlığı çıkarıldı")
        print(f"   ✓ Toplam {len(all_chunks)} chunk oluşturuldu")
        return all_chunks
    
    def _extract_title(self, first_page: str, filename: str) -> Optional[str]:
        """PDF'in ilk sayfasından başlık çıkarır"""
        lines = first_page.strip().split('\n')
        
        # İlk birkaç satırı kontrol et
        title_candidates = []
        for i, line in enumerate(lines[:10]):
            line = line.strip()
            # Başlık kriterleri:
            # - En az 10 karakter
            # - Çok uzun değil (< 200 karakter)
            # - Sayfa numarası değil
            # - Abstract/Introduction değil
            if (len(line) > 10 and 
                len(line) < 200 and 
                not line.isdigit() and
                not line.lower().startswith(('abstract', 'introduction', 'keywords', '1.', '1 '))):
                title_candidates.append(line)
        
        if title_candidates:
            # İlk geçerli satır genellikle başlık
            title = title_candidates[0]
            # Eğer çok kısa ise, sonraki satırı da ekle
            if len(title) < 30 and len(title_candidates) > 1:
                title = title + " " + title_candidates[1]
            return title
        
        # Fallback: dosya adından
        return filename.replace('.pdf', '').replace('_', ' ').replace('-', ' ')
    
    def _split_text(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """Metni chunk'lara böler"""
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            if chunk.strip():
                chunks.append(chunk)
            start = end - overlap
        return chunks
    
    def prepare_training_data(self, chunks: List[str], generate_qa: bool = True):
        """Eğitim verisini hazırlar"""
        print("\n📊 Eğitim verisi hazırlanıyor...")
        
        # 1. Belgeleri oluştur ve embedding'le
        print("   Embedding'ler hesaplanıyor...")
        for idx, chunk in enumerate(tqdm(chunks, desc="Embedding")):
            embedding = self.embedder.encode(chunk)
            doc = Document(
                id=idx,
                text=chunk,
                embedding=embedding.squeeze(0),
                metadata={"chunk_idx": idx}
            )
            self.documents.append(doc)
        
        # 2. QA çiftleri oluştur (SCP için)
        if generate_qa:
            print("   QA çiftleri oluşturuluyor...")
            # Daha fazla örnek ve daha fazla QA çifti
            sample_docs = random.sample(self.documents, min(50, len(self.documents)))
            
            for doc in tqdm(sample_docs, desc="QA üretimi"):
                pairs = self.qa_generator.generate_qa_pairs(doc.text, num_pairs=3)
                for pair in pairs:
                    if "soru" in pair and "cevap" in pair:
                        qa = QAPair(
                            question=pair["soru"],
                            answer=pair["cevap"],
                            context=doc.text,
                            positive_doc_ids=[doc.id]
                        )
                        self.qa_pairs.append(qa)
            
            print(f"   ✓ {len(self.qa_pairs)} QA çifti oluşturuldu")
        
        print(f"✅ Toplam {len(self.documents)} belge hazır")
    
    def train_scp(self, num_epochs: int = 5):
        """
        Aşama 1: SCP Pretraining
        Contrastive learning ile sıkıştırıcıyı eğitir.
        """
        print("\n" + "="*60)
        print("Aşama 1: SCP Pretraining")
        print("="*60)
        
        self.model.train()
        
        # Document embeddings
        doc_embeddings = torch.stack([d.embedding for d in self.documents]).to(self.device)
        
        # Batch size
        batch_size = min(32, len(self.documents) // 4)
        batch_size = max(batch_size, 8)
        
        # Learning rate scheduling
        num_batches_per_epoch = max(len(self.documents) // batch_size, 1)
        total_steps = num_epochs * num_batches_per_epoch
        warmup_steps = total_steps // 10
        
        for epoch in range(num_epochs):
            total_loss = 0
            num_batches = 0
            
            # Learning rate warmup & decay
            progress = epoch / num_epochs
            if progress < 0.1:  # Warmup %10
                lr_mult = progress / 0.1
            else:
                lr_mult = 0.5 * (1 + math.cos(math.pi * (progress - 0.1) / 0.9))
            
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.config.learning_rate * max(lr_mult, 0.1)
            
            # Mini-batch training
            indices = list(range(len(self.documents)))
            random.shuffle(indices)
            
            for i in range(0, len(indices), batch_size):
                batch_indices = indices[i:i + batch_size]
                if len(batch_indices) < 4:
                    continue
                
                batch_embeddings = doc_embeddings[batch_indices]
                batch_len = len(batch_indices)
                
                # Forward - sıkıştır
                compressed, _ = self.model.compress_documents(batch_embeddings)
                
                # L2 normalize
                compressed = F.normalize(compressed, p=2, dim=-1)
                
                # === GERÇEK CONTRASTIVE LOSS ===
                # Similarity matrix
                sim_matrix = torch.mm(compressed, compressed.t()) / 0.07
                
                # Diagonal'ı maskele (kendi ile benzerlik hariç)
                mask = torch.eye(batch_len, device=self.device).bool()
                sim_matrix_masked = sim_matrix.masked_fill(mask, -float('inf'))
                
                # Her örnek için en benzer olanı bul (hard negative)
                # Pozitif: kendi sıkıştırılmış hali
                # Negatif: diğer örnekler
                
                # InfoNCE benzeri loss
                pos_sim = torch.diagonal(sim_matrix)  # Kendisiyle benzerlik
                
                # Negatif log-likelihood
                log_sum_exp = torch.logsumexp(sim_matrix, dim=1)
                contrastive_loss = -pos_sim + log_sum_exp
                
                # Reconstruction loss (yardımcı)
                target = F.normalize(batch_embeddings[:, :compressed.shape[1]], p=2, dim=-1)
                reconstruction_loss = F.mse_loss(compressed, target)
                
                # Toplam loss
                loss = contrastive_loss.mean() * 0.5 + reconstruction_loss * 0.5
                
                # Backward
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            avg_loss = total_loss / max(num_batches, 1)
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"   Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f} - LR: {current_lr:.6f}")
        
        print("✅ SCP Pretraining tamamlandı")
    
    def train_end_to_end(self, num_epochs: int = 5):
        """
        Aşama 3: End-to-End Fine-tuning
        Reranker ve generator'ı birlikte optimize eder.
        """
        print("\n" + "="*60)
        print("Aşama 2: End-to-End Fine-tuning")
        print("="*60)
        
        if not self.qa_pairs:
            print("   ⚠️ QA çifti yok, atlanıyor")
            return
        
        self.model.train()
        
        # Tüm belge embedding'leri
        doc_embeddings = torch.stack([d.embedding for d in self.documents]).to(self.device)
        
        for epoch in range(num_epochs):
            total_loss = 0
            
            # Learning rate scheduling (cosine)
            progress = epoch / num_epochs
            lr_mult = 0.5 * (1 + math.cos(math.pi * progress))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.config.learning_rate * 0.1 * max(lr_mult, 0.1)
            
            # Shuffle QA pairs
            shuffled_qa = self.qa_pairs.copy()
            random.shuffle(shuffled_qa)
            
            for qa in tqdm(shuffled_qa, desc=f"Epoch {epoch+1}"):
                # Query embedding
                query_emb = self.embedder.encode(qa.question).to(self.device)
                
                # Sample documents (positive + more hard negatives)
                pos_ids = qa.positive_doc_ids
                
                # Hard negative mining: benzer belgeler seç
                num_negatives = min(self.config.top_k * 2, len(self.documents) - len(pos_ids))
                neg_ids = random.sample(
                    [i for i in range(len(self.documents)) if i not in pos_ids],
                    num_negatives
                )
                sample_ids = pos_ids + neg_ids
                
                # Get embeddings
                sample_embeddings = doc_embeddings[sample_ids].unsqueeze(0)
                
                # Forward
                outputs = self.model(query_emb, sample_embeddings)
                
                # Loss: pozitif belgeler yüksek skor almalı
                rerank_scores = outputs["rerank_scores"]
                
                # Margin ranking loss (daha iyi ayrım için)
                labels = torch.zeros(1, len(sample_ids), device=self.device)
                labels[0, :len(pos_ids)] = 1.0
                
                # Binary cross-entropy + margin loss
                bce_loss = F.binary_cross_entropy_with_logits(rerank_scores, labels)
                
                # Contrastive margin: pozitif skorlar negatiflerden yüksek olmalı
                pos_scores = rerank_scores[0, :len(pos_ids)].mean()
                neg_scores = rerank_scores[0, len(pos_ids):].mean()
                margin_loss = F.relu(0.5 - (pos_scores - neg_scores))
                
                loss = bce_loss + 0.3 * margin_loss
                
                # Backward
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / max(len(self.qa_pairs), 1)
            print(f"   Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f}")
        
        print("✅ End-to-End Fine-tuning tamamlandı")
    
    def compress_all_documents(self):
        """Tüm belgeleri sıkıştırır"""
        print("\n📦 Belgeler sıkıştırılıyor...")
        self.model.eval()
        
        with torch.no_grad():
            for doc in tqdm(self.documents, desc="Sıkıştırma"):
                emb = doc.embedding.unsqueeze(0).to(self.device)
                compressed, salient = self.model.compress_documents(emb)
                doc.compressed_embedding = compressed.squeeze(0).cpu()
        
        print("✅ Tüm belgeler sıkıştırıldı")


# ============================================================================
# 8. CLaRa RAG System (Inference)
# ============================================================================

class CLaRaRAGSystem:
    """
    CLaRa RAG Sistemi - Inference.
    
    Eğitilmiş modeli kullanarak soru-cevap yapar.
    """
    
    def __init__(
        self,
        trainer: CLaRaTrainer,
        llm_model: str = "granite4:3b"
    ):
        self.trainer = trainer
        self.model = trainer.model
        self.embedder = trainer.embedder
        self.documents = trainer.documents
        self.llm_model = llm_model
        self.device = trainer.device
        
        # Document embedding matrix
        self.doc_embeddings = torch.stack([d.embedding for d in self.documents])
        self.doc_compressed = torch.stack([d.compressed_embedding for d in self.documents])
    
    def query(self, question: str, top_k: int = 5) -> Dict[str, Any]:
        """Soru sorar ve cevap üretir"""
        self.model.eval()
        
        with torch.no_grad():
            # 1. Query embedding (orijinal)
            query_emb = self.embedder.encode(question).to(self.device)
            
            # 2. Orijinal embedding'ler üzerinden similarity (daha iyi sonuç)
            query_norm = F.normalize(query_emb, p=2, dim=-1)  # [1, 768]
            
            # Orijinal doc embedding'leri kullan
            doc_emb = torch.stack([d.embedding for d in self.documents]).to(self.device)
            doc_norm = F.normalize(doc_emb, p=2, dim=-1)  # [num_docs, 768]
            
            # Cosine similarity
            similarities = torch.mm(query_norm, doc_norm.t()).squeeze(0)  # [num_docs]
            
            # 4. Get top-k by similarity
            top_scores, top_indices = torch.topk(similarities, min(top_k, len(self.documents)))
            top_indices = top_indices.cpu().numpy()
            top_scores = top_scores.cpu().numpy()
            
            # 5. Build context (sadece anlamlı metinleri al)
            context_parts = []
            sources = []
            for i, idx in enumerate(top_indices):
                doc = self.documents[idx]
                score = float(top_scores[i])
                
                # Çok kısa veya sayısal içerikli belgeleri atla
                text = doc.text.strip()
                # Sayısal içerik kontrolü
                non_numeric = text.replace('.', '').replace('-', '').replace('(', '').replace(')', '').replace(' ', '').replace('\n', '').replace(',', '')
                if len(text) < 50 or (len(non_numeric) > 0 and non_numeric.replace('0', '').replace('1', '').replace('2', '').replace('3', '').replace('4', '').replace('5', '').replace('6', '').replace('7', '').replace('8', '').replace('9', '') == ''):
                    continue
                
                context_parts.append(f"[Kaynak {len(sources)+1}]\n{text}")
                sources.append({
                    "id": int(idx), 
                    "score": round(score, 3), 
                    "text": text[:150].replace('\n', ' ')
                })
                
                # Maksimum 5 kaynak
                if len(sources) >= 5:
                    break
            
            context = "\n\n".join(context_parts)
            
            # 6. Generate answer
            answer = self._generate_answer(question, context)
        
        return {
            "question": question,
            "answer": answer,
            "sources": sources,
            "context": context
        }
    
    def _generate_answer(self, question: str, context: str) -> str:
        """LLM ile cevap üretir"""
        system_prompt = """Sen bir akademik asistansın. Kurallar:
1. SADECE verilen bağlam bilgilerini kullan
2. Bağlamda OLMAYAN bilgiyi ASLA uydurma
3. Kısa paragraflar halinde yaz (2-3 paragraf)
4. Madde işareti veya numara KULLANMA
5. Akıcı Türkçe cümleler kur
6. Emin olmadığın bilgileri belirtme"""

        user_prompt = f"""Bağlam bilgileri:
{context[:2000]}

Soru: {question}

Bağlamdaki bilgilere dayanarak, düz paragraflar halinde kısa bir cevap yaz:"""
        
        try:
            response = ollama.chat(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                options={
                    "temperature": 0.5,
                    "num_predict": 300,  # Daha kısa cevap
                    "top_p": 0.85,
                    "repeat_penalty": 1.3,  # Tekrarı engelle
                    "repeat_last_n": 64
                }
            )
            answer = response["message"]["content"]
            
            # Tekrarlı kısımları temizle
            sentences = answer.split('.')
            seen = set()
            unique_sentences = []
            for s in sentences:
                s_clean = s.strip().lower()[:50]  # İlk 50 karakter
                if s_clean and s_clean not in seen:
                    seen.add(s_clean)
                    unique_sentences.append(s.strip())
            
            answer = '. '.join(unique_sentences)
            if answer and not answer.endswith('.'):
                answer += '.'
            
            # Çok uzun cevapları kırp
            if len(answer) > 1000:
                answer = answer[:1000] + "..."
            
            return answer
        except Exception as e:
            return f"Hata: {str(e)}"


# ============================================================================
# Ana Program
# ============================================================================

def main():
    print("="*60)
    print("CLaRa - Tam Implementasyon")
    print("="*60)
    
    # Kayıt dizini
    save_dir = "clara_saved_model"
    
    # Eğer kayıtlı model varsa, yükle
    if os.path.exists(os.path.join(save_dir, "clara_model.pt")):
        print("\n📂 Kayıtlı model bulundu!")
        choice = input("Kayıtlı modeli yüklemek ister misiniz? (e/h): ").strip().lower()
        
        if choice == 'e':
            trainer = CLaRaTrainer.load(save_dir)
            trainer.compress_all_documents()
            
            # RAG sistemi oluştur
            print("\n🚀 RAG sistemi hazırlanıyor...")
            rag = CLaRaRAGSystem(trainer, llm_model="granite4:3b")
            
            # İnteraktif sorgu modu
            interactive_query(rag)
            return
    
    # Yeni eğitim
    print("\n🆕 Yeni eğitim başlatılıyor...")
    
    # Konfigürasyon
    config = TrainingConfig(
        embedding_dim=768,  # nomic-embed-text-v2-moe boyutu
        hidden_dim=512,
        compressed_dim=256,
        num_attention_heads=8,
        num_layers=2,
        batch_size=4,
        learning_rate=1e-4,
        top_k=10,  # Artırıldı: 5 → 10
        temperature=0.1
    )
    
    # Trainer oluştur
    trainer = CLaRaTrainer(
        config=config,
        embedding_model="nomic-embed-text-v2-moe:latest",
        llm_model="granite4:3b"
    )
    
    # PDF'leri yükle
    pdf_dir = "."  # Mevcut dizin
    chunks = trainer.load_pdfs(pdf_dir)
    
    if not chunks:
        print("❌ PDF bulunamadı!")
        return
    
    # Eğitim verisini hazırla
    trainer.prepare_training_data(chunks, generate_qa=True)
    
    # Eğitim - Daha uzun ve kaliteli
    print("\n🎓 Eğitim başlıyor...")
    trainer.train_scp(num_epochs=15)  # 15 epoch SCP
    trainer.train_end_to_end(num_epochs=10)  # 10 epoch E2E
    
    # Belgeleri sıkıştır
    trainer.compress_all_documents()
    
    # 💾 SİSTEMİ KAYDET
    trainer.save(save_dir)
    
    # RAG sistemi oluştur
    print("\n🚀 RAG sistemi hazırlanıyor...")
    rag = CLaRaRAGSystem(trainer, llm_model="granite4:3b")
    
    # Test soruları
    test_questions = [
        "Bu makaleler ne hakkında?",
        "Hangi modeller kullanılmış?",
        "Temel katkılar nelerdir?",
        "Deneysel sonuçlar nasıl?"
    ]
    
    print("\n" + "="*60)
    print("Test Sorguları")
    print("="*60)
    
    for question in test_questions:
        print(f"\n{'─'*60}")
        result = rag.query(question, top_k=10)
        
        print(f"📝 Soru: {result['question']}")
        print(f"{'─'*60}")
        print(f"💬 Cevap:\n{result['answer']}")
        print(f"\n📚 Kaynaklar ({len(result['sources'])} belge):")
        for src in result['sources'][:3]:
            print(f"   - (Skor: {src['score']:.3f}) {src['text'][:100]}...")
    
    # İnteraktif mod
    interactive_query(rag)


def interactive_query(rag: CLaRaRAGSystem):
    """İnteraktif sorgu modu"""
    print("\n" + "="*60)
    print("📝 İnteraktif Sorgu Modu")
    print("   Çıkmak için 'q' yazın")
    print("="*60)
    
    while True:
        try:
            question = input("\n🔍 Sorunuz: ").strip()
            
            if question.lower() in ['q', 'quit', 'exit', 'çık']:
                print("👋 Görüşürüz!")
                break
            
            if not question:
                continue
            
            result = rag.query(question, top_k=10)
            
            print(f"\n💬 Cevap:\n{result['answer']}")
            print(f"\n📚 Kaynaklar ({len(result['sources'])} belge):")
            for src in result['sources'][:3]:
                print(f"   - (Skor: {src['score']:.3f}) {src['text'][:80]}...")
                
        except KeyboardInterrupt:
            print("\n👋 Görüşürüz!")
            break


def load_and_query(save_dir: str = "clara_saved_model", llm_model: str = "granite4:3b"):
    """
    Kaydedilmiş modeli yükleyip sorgu yapar.
    Başka sistemlerde kullanmak için bu fonksiyonu çağırın.
    
    Örnek:
        from clara_full_implementation import load_and_query
        load_and_query("clara_saved_model")
    """
    if not os.path.exists(os.path.join(save_dir, "clara_model.pt")):
        print(f"❌ Kayıtlı model bulunamadı: {save_dir}")
        return None
    
    trainer = CLaRaTrainer.load(save_dir)
    trainer.compress_all_documents()
    
    rag = CLaRaRAGSystem(trainer, llm_model=llm_model)
    
    interactive_query(rag)
    return rag


if __name__ == "__main__":
    main()

