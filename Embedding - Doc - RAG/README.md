# CLaRa-Inspired Unified RAG Sistemi

Apple'ın **CLaRa (Continuous Latent Reasoning)** yaklaşımından esinlenen, retrieval ve generation süreçlerini birleşik sürekli uzayda optimize eden RAG sistemi.

**🔗 Tamamen Ollama Tabanlı**

## 🌟 Özellikler

| Özellik | Açıklama |
|---------|----------|
| 🔄 **Birleşik Optimizasyon** | Retrieval ve generation aynı uzayda çalışır |
| 📦 **Belge Sıkıştırma** | Salient bilgiler korunarak vektörler sıkıştırılır |
| 🎯 **Soft Reranking** | Differentiable sıralama ile gradyan aktarımı |
| 🧠 **End-to-End** | Tüm bileşenler birlikte optimize edilir |
| 🔗 **Ollama** | Hem embedding hem LLM için Ollama kullanır |

## 📋 Gereksinimler

```bash
pip install -r requirements.txt
```

**Desteklenen Ollama Modelleri:**
- **Embedding:** `nomic-embed-text-v2-moe:latest`, `embeddinggemma:latest`
- **LLM:** `granite4:3b`, `rnj-1:latest`, `deepseek-ocr:latest`

## 🚀 Kullanım

### Python API

```python
from clara_unified_rag import CLaRaRAG

# Sistem oluştur (Tamamen Ollama tabanlı)
clara = CLaRaRAG(
    embedding_model_name="nomic-embed-text-v2-moe:latest",
    llm_model_name="granite4:3b"
)

# Belgeleri ekle
belgeler = [
    "Yapay zeka, insan zekasını taklit eden sistemlerdir.",
    "Derin öğrenme, sinir ağları kullanır.",
    "NLP, bilgisayarların dili anlamasını sağlar."
]
clara.add_documents(belgeler)

# Soru sor
result = clara.query("Yapay zeka nedir?")
print(result.answer)
print(f"Güvenilirlik: {result.confidence_score:.2f}")
```

### Gradio Arayüzü

```bash
python clara_unified_rag.py
```

Tarayıcıda `http://localhost:7860` adresine gidin.

## 🏗️ Mimari

```
┌─────────────┐     ┌──────────────────┐     ┌─────────────┐
│   Belgeler  │────▶│ DocumentCompressor│────▶│  Embedding  │
└─────────────┘     └──────────────────┘     │    Space    │
                                              └──────┬──────┘
┌─────────────┐                                      │
│    Sorgu    │──────────────────────────────────────┤
└─────────────┘                                      ▼
                                              ┌─────────────┐
                                              │   Search    │
                                              └──────┬──────┘
                                                     │
                                              ┌──────▼──────┐
                                              │  Reranker   │
                                              └──────┬──────┘
                                                     │
                                              ┌──────▼──────┐
                                              │  Generator  │
                                              └──────┬──────┘
                                                     │
                                              ┌──────▼──────┐
                                              │   Cevap     │
                                              └─────────────┘
```

## 📊 Bileşenler

### 1. DocumentCompressor
- TF-IDF tabanlı önem ağırlıklandırması
- Salient (önemli) ifade çıkarma
- Türkçe stop words desteği

### 2. UnifiedEmbeddingSpace
- Sorgu ve belgeler aynı uzayda
- Cosine similarity ile hızlı arama

### 3. SoftReranker
- Attention tabanlı skor hesaplama
- Differentiable yapı

### 4. Generator
- Ollama entegrasyonu
- Türkçe prompt desteği

## 📝 Klasik RAG vs CLaRa

| Özellik | Klasik RAG | CLaRa |
|---------|------------|-------|
| Optimizasyon | Ayrık | Birleşik |
| Belge Seçimi | Sadece similarity | Similarity + Reranking |
| Gradyan Aktarımı | Yok | Var (Soft Reranking) |
| Bağlam Kullanımı | Sabit | Dinamik ağırlıklı |

## 📄 Lisans

MIT License

