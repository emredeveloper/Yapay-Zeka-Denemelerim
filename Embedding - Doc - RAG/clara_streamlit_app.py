"""
CLaRa RAG - Streamlit Arayüzü
==============================
Kaydedilmiş CLaRa modelini kullanarak soru-cevap arayüzü.
"""

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import json
import os
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, field
import ollama

# ============================================================================
# Sayfa Konfigürasyonu
# ============================================================================
st.set_page_config(
    page_title="CLaRa RAG",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CSS Stilleri
# ============================================================================
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    }
    
    .main-title {
        font-family: 'Segoe UI', sans-serif;
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(90deg, #00d4ff, #7b2cbf, #ff6b6b);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
    }
    
    .sub-title {
        text-align: center;
        color: #a0a0a0;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    .answer-box {
        background: linear-gradient(145deg, #1f4037, #162447);
        border-left: 4px solid #00ff88;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        color: #e0e0e0;
        line-height: 1.8;
    }
    
    .stats-box {
        background: linear-gradient(145deg, #2d2d44, #1a1a2e);
        border-radius: 15px;
        padding: 1rem;
        text-align: center;
        border: 1px solid #4a4a6a;
    }
    
    .stats-number {
        font-size: 2rem;
        font-weight: 700;
        color: #00d4ff;
    }
    
    .stats-label {
        color: #a0a0a0;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)


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
class TrainingConfig:
    """Eğitim konfigürasyonu"""
    embedding_dim: int = 768
    hidden_dim: int = 512
    compressed_dim: int = 256
    num_attention_heads: int = 8
    num_layers: int = 2
    batch_size: int = 8
    learning_rate: float = 1e-4
    num_epochs: int = 10
    warmup_steps: int = 100
    top_k: int = 5
    temperature: float = 0.1
    contrastive_weight: float = 1.0
    generation_weight: float = 1.0
    rerank_weight: float = 0.5


# ============================================================================
# Neural Network Bileşenleri (Orijinal mimariye uygun)
# ============================================================================

class DocumentCompressorNetwork(nn.Module):
    """Belge Sıkıştırma Ağı - Orijinal yapıya uygun"""
    
    def __init__(self, config: TrainingConfig):
        super().__init__()
        self.config = config
        
        self.input_proj = nn.Linear(config.embedding_dim, config.hidden_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_attention_heads,
            dim_feedforward=config.hidden_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        
        self.compress_layers = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_dim, config.compressed_dim)
        )
        
        self.salient_attention = nn.MultiheadAttention(
            embed_dim=config.hidden_dim,
            num_heads=config.num_attention_heads,
            batch_first=True
        )
        
        self.salient_query = nn.Parameter(torch.randn(1, 1, config.hidden_dim))
    
    def forward(self, embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if embeddings.dim() == 2:
            embeddings = embeddings.unsqueeze(1)
        
        batch_size = embeddings.shape[0]
        x = self.input_proj(embeddings)
        x = self.transformer(x)
        
        query = self.salient_query.expand(batch_size, -1, -1)
        attended, salient_weights = self.salient_attention(query, x, x)
        salient_weights = salient_weights.squeeze(1)
        
        weighted_x = x * salient_weights.unsqueeze(-1)
        pooled = weighted_x.sum(dim=1)
        
        compressed = self.compress_layers(pooled)
        compressed = F.normalize(compressed, p=2, dim=-1)
        
        return compressed, salient_weights


class DifferentiableTopK(nn.Module):
    """Differentiable Top-K"""
    
    def __init__(self, k: int = 5, temperature: float = 0.1):
        super().__init__()
        self.k = k
        self.temperature = temperature
    
    def forward(self, scores: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_docs = scores.shape
        scaled_scores = scores / self.temperature
        soft_selection = F.softmax(scaled_scores, dim=-1)
        _, hard_indices = torch.topk(scores, min(self.k, num_docs), dim=-1)
        hard_selection = torch.zeros_like(soft_selection)
        hard_selection.scatter_(1, hard_indices, 1.0)
        selection = hard_selection - soft_selection.detach() + soft_selection
        return selection, hard_indices


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
    
    def forward(self, query: torch.Tensor, doc_compressed: torch.Tensor, initial_scores: torch.Tensor) -> torch.Tensor:
        batch_size, num_docs, _ = doc_compressed.shape
        q = self.query_proj(query).unsqueeze(1)
        d = self.doc_proj(doc_compressed)
        attended, _ = self.cross_attention(q, d, d)
        interaction = d * attended
        scores = self.score_head(interaction).squeeze(-1)
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
    
    def forward(self, doc_compressed: torch.Tensor, rerank_scores: torch.Tensor) -> torch.Tensor:
        weights = F.softmax(rerank_scores, dim=-1).unsqueeze(-1)
        fused = (doc_compressed * weights).sum(dim=1)
        fused = self.output_proj(fused)
        return fused


class JointRerankerGenerator(nn.Module):
    """Birleşik Reranker ve Generator modülü"""
    
    def __init__(self, config: TrainingConfig):
        super().__init__()
        self.config = config
        self.reranker = RerankerModule(config)
        self.context_fusion = ContextFusionModule(config)
        self.generation_scorer = nn.Sequential(
            nn.Linear(config.compressed_dim * 2, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_dim, 1)
        )
    
    def forward(self, query_embedding, doc_embeddings, doc_compressed, initial_scores):
        batch_size, num_docs, _ = doc_embeddings.shape
        rerank_scores = self.reranker(query_embedding, doc_compressed, initial_scores)
        fused_context = self.context_fusion(doc_compressed, rerank_scores)
        query_compressed = query_embedding[:, :self.config.compressed_dim]
        combined = torch.cat([query_compressed, fused_context], dim=-1)
        generation_scores = self.generation_scorer(combined).squeeze(-1)
        return rerank_scores, fused_context, generation_scores


class CLaRaModel(nn.Module):
    """Tam CLaRa Modeli - Orijinal yapıya uygun"""
    
    def __init__(self, config: TrainingConfig):
        super().__init__()
        self.config = config
        
        self.compressor = DocumentCompressorNetwork(config)
        self.topk_selector = DifferentiableTopK(k=config.top_k, temperature=config.temperature)
        self.reranker_generator = JointRerankerGenerator(config)
        
        self.query_encoder = nn.Sequential(
            nn.Linear(config.embedding_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.compressed_dim)
        )
    
    def compress_documents(self, doc_embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.compressor(doc_embeddings)
    
    def encode_query(self, query_embedding: torch.Tensor) -> torch.Tensor:
        return self.query_encoder(query_embedding)


# ============================================================================
# Ollama Bileşenleri
# ============================================================================

class OllamaEmbeddings:
    """Ollama embedding modeli"""
    
    def __init__(self, model_name: str = "nomic-embed-text-v2-moe:latest"):
        self.model_name = model_name
        self._dim = None
    
    def encode(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = []
        for text in texts:
            truncated = text[:2000] if len(text) > 2000 else text
            try:
                response = ollama.embeddings(model=self.model_name, prompt=truncated)
                embeddings.append(response["embedding"])
            except:
                try:
                    response = ollama.embeddings(model=self.model_name, prompt=text[:500])
                    embeddings.append(response["embedding"])
                except:
                    if self._dim:
                        embeddings.append([0.0] * self._dim)
                    else:
                        raise
        
        return torch.tensor(embeddings, dtype=torch.float32)
    
    @property
    def embedding_dim(self) -> int:
        if self._dim is None:
            test = self.encode("test")
            self._dim = test.shape[-1]
        return self._dim


# ============================================================================
# CLaRa RAG Sistemi
# ============================================================================

class CLaRaRAG:
    """CLaRa RAG Sistemi"""
    
    def __init__(self, model_path: str, llm_model: str = "granite4:3b"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.llm_model = llm_model
        self.documents: List[Document] = []
        self._load_model(model_path)
    
    def _load_model(self, model_path: str):
        """Kaydedilmiş modeli yükle"""
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
        
        self.embedder = OllamaEmbeddings(config_dict.get("embedding_model", "nomic-embed-text-v2-moe:latest"))
        self.llm_model = config_dict.get("llm_model", "granite4:3b")
        
        self.model = CLaRaModel(self.config).to(self.device)
        
        model_file = os.path.join(model_path, "clara_model.pt")
        if os.path.exists(model_file):
            checkpoint = torch.load(model_file, map_location=self.device, weights_only=False)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
            else:
                self.model.load_state_dict(checkpoint)
        
        self.model.eval()
        
        docs_path = os.path.join(model_path, "documents.pkl")
        if os.path.exists(docs_path):
            with open(docs_path, "rb") as f:
                loaded_docs = pickle.load(f)
                # Dict ise Document'a dönüştür
                self.documents = []
                for doc in loaded_docs:
                    if isinstance(doc, dict):
                        d = Document(
                            id=doc.get("id", 0),
                            text=doc.get("text", ""),
                            embedding=doc.get("embedding"),
                            compressed_embedding=doc.get("compressed_embedding"),
                            salient_tokens=doc.get("salient_tokens", []),
                            metadata=doc.get("metadata", {})
                        )
                        self.documents.append(d)
                    else:
                        self.documents.append(doc)
    
    def _keyword_score(self, query: str, text: str) -> float:
        """Basit keyword matching skoru (BM25 benzeri)"""
        query_lower = query.lower()
        text_lower = text.lower()
        
        # === ÖZEL: Başlık chunk'ları için yüksek skor ===
        if "[makale başlığı]" in text_lower:
            if any(kw in query_lower for kw in ["başlık", "title", "isim", "adı", "ne"]):
                return 1.0  # Maksimum skor
        
        # Query'den anahtar kelimeler çıkar
        keywords = [w for w in query_lower.split() if len(w) > 2]
        
        # Özel anahtar kelimeler ekle (Türkçe-İngilizce mapping)
        keyword_map = {
            "başlık": ["title", "titled", "başlık", "[makale başlığı]"],
            "makale": ["paper", "article", "study", "research", "makale"],
            "model": ["model", "bert", "gpt", "transformer", "llm", "gemma"],
            "sonuç": ["result", "conclusion", "finding", "sonuç"],
            "deney": ["experiment", "evaluation", "test", "deney"],
            "katkı": ["contribution", "propose", "introduce", "katkı"],
            "nedir": ["what", "is", "are", "nedir", "ne"],
        }
        
        # Expanded keywords
        expanded = set(keywords)
        for kw in keywords:
            for key, values in keyword_map.items():
                if key in kw or kw in key:
                    expanded.update(values)
        
        # Skor hesapla
        score = 0.0
        for kw in expanded:
            if kw in text_lower:
                count = text_lower.count(kw)
                score += min(count * 0.1, 0.3)
        
        return min(score, 1.0)
    
    def query(self, question: str, top_k: int = 5) -> Dict[str, Any]:
        """Soru sor ve cevap al - Hybrid Search (Embedding + Keyword)"""
        if not self.documents:
            return {"answer": "Hiç belge yüklenmemiş!", "sources": [], "scores": []}
        
        with torch.no_grad():
            query_emb = self.embedder.encode(question)[0].to(self.device)
            query_compressed, _ = self.model.compressor(query_emb.unsqueeze(0))
            query_compressed = query_compressed.squeeze(0)
            
            doc_scores = []
            for doc in self.documents:
                # === 1. EMBEDDING SKORU ===
                embedding_score = 0.0
                orig_emb = doc.embedding
                if orig_emb is not None:
                    if not isinstance(orig_emb, torch.Tensor):
                        orig_emb = torch.tensor(orig_emb, dtype=torch.float32)
                    orig_emb = orig_emb.to(self.device)
                    
                    if orig_emb.dim() == 1:
                        orig_emb = orig_emb.unsqueeze(0)
                    comp_emb, _ = self.model.compressor(orig_emb)
                    comp_emb = comp_emb.squeeze(0)
                    
                    embedding_score = F.cosine_similarity(
                        query_compressed.unsqueeze(0), 
                        comp_emb.unsqueeze(0)
                    ).item()
                elif doc.compressed_embedding is not None:
                    comp_emb = doc.compressed_embedding
                    if not isinstance(comp_emb, torch.Tensor):
                        comp_emb = torch.tensor(comp_emb, dtype=torch.float32)
                    comp_emb = comp_emb.to(self.device)
                    
                    if comp_emb.dim() == 1:
                        comp_emb = comp_emb.unsqueeze(0)
                    
                    embedding_score = F.cosine_similarity(
                        query_compressed.unsqueeze(0), 
                        comp_emb
                    ).item()
                
                # === 2. KEYWORD SKORU ===
                keyword_score = self._keyword_score(question, doc.text)
                
                # === 3. HYBRID SKOR ===
                # %70 embedding + %30 keyword
                hybrid_score = 0.7 * embedding_score + 0.3 * keyword_score
                
                doc_scores.append((doc, hybrid_score, embedding_score, keyword_score))
            
            # Hybrid skora göre sırala
            doc_scores.sort(key=lambda x: x[1], reverse=True)
            top_docs = doc_scores[:min(top_k * 2, len(doc_scores))]
            
            if not top_docs:
                return {"answer": "İlgili belge bulunamadı.", "sources": [], "scores": []}
            
            final_docs = top_docs[:top_k]
            
            context_parts = []
            for doc, hybrid, emb, kw in final_docs:
                text = doc.text.strip()
                if len(text) > 30:
                    context_parts.append(f"[Skor: {hybrid:.3f} | Emb: {emb:.2f} | KW: {kw:.2f}]\n{text}")
            
            # 5 belge al
            context = "\n\n---\n\n".join(context_parts[:5])
            answer = self._generate_answer(question, context)
            
            return {
                "answer": answer,
                "sources": [(doc.text[:200], hybrid) for doc, hybrid, _, _ in final_docs[:5]],
                "scores": [hybrid for _, hybrid, _, _ in final_docs[:5]]
            }
    
    def _generate_answer(self, question: str, context: str) -> str:
        """LLM ile cevap üret"""
        prompt = f"""Sen bir araştırma asistanısın. Aşağıdaki bağlam bilgilerini kullanarak soruyu yanıtla.

KURALLAR:
- Bağlamdaki bilgileri kullanarak cevap ver
- Başlık sorulursa, bağlamda geçen makale/çalışma isimlerini listele
- Teknik terimler ve başlıklar orijinal dilde (İngilizce) kalabilir
- Bağlamda doğrudan bilgi yoksa, mevcut bilgilerden çıkarım yap
- Türkçe ve akıcı cümleler kur

BAĞLAM:
{context[:5000]}

SORU: {question}

CEVAP:"""
        
        try:
            response = ollama.generate(
                model=self.llm_model,
                prompt=prompt,
                options={
                    "temperature": 0.3,
                    "num_predict": 400,
                    "top_p": 0.9
                }
            )
            return response["response"].strip()
        except Exception as e:
            return f"Cevap üretilirken hata: {str(e)}"


# ============================================================================
# Streamlit Arayüzü
# ============================================================================

@st.cache_resource
def load_clara_system():
    """CLaRa sistemini yükle"""
    model_path = os.path.join(os.path.dirname(__file__), "clara_saved_model")
    if not os.path.exists(model_path):
        return None
    return CLaRaRAG(model_path)


def main():
    st.markdown('<h1 class="main-title">🧠 CLaRa RAG</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Continuous Latent Reasoning ile Akıllı Belge Sorgulama</p>', unsafe_allow_html=True)
    
    with st.spinner("🔄 CLaRa sistemi yükleniyor..."):
        clara = load_clara_system()
    
    if clara is None:
        st.error("❌ Model bulunamadı! Önce `clara_full_implementation.py` ile modeli eğitin.")
        st.info("📁 Beklenen konum: `clara_saved_model/`")
        return
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 📊 Sistem Bilgileri")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class="stats-box">
                <div class="stats-number">{len(clara.documents)}</div>
                <div class="stats-label">Belge</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="stats-box">
                <div class="stats-number">{clara.config.compressed_dim}</div>
                <div class="stats-label">Boyut</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### ⚙️ Ayarlar")
        top_k = st.slider("📚 Kaynak Sayısı", min_value=1, max_value=10, value=5)
        
        st.markdown("---")
        st.markdown("### 🤖 Model Bilgisi")
        st.info(f"**LLM:** {clara.llm_model}")
        st.info(f"**Embedding:** nomic-embed-text-v2-moe")
        
        st.markdown("---")
        if st.button("🔄 Sistemi Yenile", use_container_width=True):
            st.cache_resource.clear()
            st.rerun()
    
    # Ana içerik
    st.markdown("### 💬 Soru Sorun")
    
    question = st.text_input(
        "Sorunuzu yazın:",
        placeholder="Örn: Bu makaleler ne hakkında?",
        label_visibility="collapsed"
    )
    
    st.markdown("**💡 Örnek Sorular:**")
    example_cols = st.columns(4)
    examples = [
        "Bu makaleler ne hakkında?",
        "Hangi modeller kullanılmış?",
        "Temel katkılar nelerdir?",
        "Gemma Scope 2 nedir?"
    ]
    
    selected_example = None
    for i, col in enumerate(example_cols):
        with col:
            if st.button(examples[i], key=f"ex_{i}", use_container_width=True):
                selected_example = examples[i]
    
    if selected_example:
        question = selected_example
    
    if question:
        st.markdown("---")
        
        with st.spinner("🔍 Aranıyor ve cevap üretiliyor..."):
            result = clara.query(question, top_k=top_k)
        
        st.markdown("### 💬 Cevap")
        st.markdown(f"""
        <div class="answer-box">
            {result['answer']}
        </div>
        """, unsafe_allow_html=True)
        
        if result['sources']:
            st.markdown("### 📚 Kaynaklar")
            
            for i, (source_text, score) in enumerate(result['sources'][:5]):
                with st.expander(f"📄 Kaynak {i+1} (Skor: {score:.3f})"):
                    st.markdown(f"```\n{source_text}...\n```")
    
    st.markdown("---")
    st.markdown(
        "<p style='text-align: center; color: #666;'>🧠 CLaRa RAG - Apple CLaRa Implementasyonu</p>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
