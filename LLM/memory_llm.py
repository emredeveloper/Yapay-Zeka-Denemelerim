import os
import json
import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime

from ollama import Client

# .env dosyasından environment variable'ları yükle
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # .env dosyası isteğe bağlı

# ---- Uzun Vadeli Bellek: vektör DB + epizodik log ----
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer


# =========================
# Logging Sistemi
# =========================
class Logger:
    """Detaylı adım adım logging"""
    def __init__(self):
        self.logs = []
    
    def log(self, step: str, status: str, message: str, data: Dict[str, Any] = None):
        """
        Adım bilgisini logla
        status: "info", "processing", "success", "warning", "error"
        """
        entry = {
            "timestamp": datetime.now().isoformat(),
            "step": step,
            "status": status,
            "message": message,
            "data": data or {}
        }
        self.logs.append(entry)
        print(f"[{status.upper()}] {step}: {message}")
    
    def get_logs(self):
        return self.logs
    
    def clear(self):
        self.logs = []


# =========================
# Config
# =========================
OLLAMA_API_URL = "https://ollama.com"
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY")
MODEL_ID = "qwen3-vl:235b-cloud"

SHORT_TERM_TOKEN_LIMIT = 8192
CHROMA_DIR = "./chroma_store"
COLL_NAME = "ltm_semantic"
EPISODIC_LOG = "episodic_log.jsonl"


# =========================
# Yardımcılar
# =========================
def ensure_jsonl(file_path: str):
    if not os.path.exists(file_path):
        with open(file_path, "w", encoding="utf-8"):
            pass

def now_ts():
    return time.strftime("%Y-%m-%d %H:%M:%S")


# =========================
# Kısa Vadeli Bellek
# =========================
@dataclass
class ShortTermMemory:
    token_limit: int = SHORT_TERM_TOKEN_LIMIT
    messages: List[Dict[str, Any]] = field(default_factory=list)

    def add(self, role: str, text: str):
        content = [{"type": "text", "text": text}]
        self.messages.append({"role": role, "content": content})
        while len(str(self.messages)) > self.token_limit * 4 and len(self.messages) > 1:
            self.messages.pop(0)

    def as_chat(self) -> List[Dict[str, str]]:
        result = []
        for msg in self.messages:
            role = msg["role"]
            text = ""
            if isinstance(msg["content"], list):
                for item in msg["content"]:
                    if item.get("type") == "text":
                        text += item["text"]
            else:
                text = msg["content"]
            result.append({"role": role, "content": text})
        return result
    
    def get_summary(self) -> Dict[str, Any]:
        """Bellek özeti"""
        return {
            "message_count": len(self.messages),
            "size_estimate": len(str(self.messages)),
            "limit": self.token_limit * 4,
            "messages": self.as_chat()
        }


# =========================
# Uzun Vadeli Bellek (Anlamsal)
# =========================
class LongTermMemory:
    def __init__(self, coll_name: str = COLL_NAME, persist_dir: str = CHROMA_DIR):
        os.makedirs(persist_dir, exist_ok=True)
        self.client = chromadb.PersistentClient(path=persist_dir, settings=Settings(allow_reset=False))
        try:
            self.coll = self.client.get_collection(coll_name)
        except Exception:
            self.coll = self.client.create_collection(coll_name)
        self.embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.logger = Logger()

    def upsert(self, docs: List[str], metadatas: Optional[List[Dict[str, Any]]] = None, ids: Optional[List[str]] = None):
        embeddings = self.embedder.encode(docs, normalize_embeddings=True).tolist()
        if ids is None:
            ids = [f"doc-{int(time.time()*1e6)}-{i}" for i in range(len(docs))]
        self.coll.upsert(ids=ids, documents=docs, metadatas=metadatas, embeddings=embeddings)
        self.logger.log("LTM:Upsert", "success", f"{len(docs)} doküman uzun vadeli belleğe eklendi")

    def query(self, q: str, k: int = 5) -> List[Dict[str, Any]]:
        q_emb = self.embedder.encode([q], normalize_embeddings=True).tolist()
        res = self.coll.query(query_embeddings=q_emb, n_results=k)
        out = []
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        for doc, meta in zip(docs, metas):
            out.append({"text": doc, "meta": meta})
        return out
    
    def count(self) -> int:
        """Bellekteki toplam doküman sayısı"""
        return self.coll.count()
    
    def get_summary(self) -> Dict[str, Any]:
        """Bellek özeti"""
        return {
            "document_count": self.count(),
            "embedder": "all-MiniLM-L6-v2"
        }


# =========================
# Epizodik Log
# =========================
class EpisodicLog:
    def __init__(self, path: str = EPISODIC_LOG):
        self.path = path
        ensure_jsonl(self.path)

    def append(self, user: str, query: str, assistant: str, context: Dict[str, Any]):
        rec = {
            "ts": now_ts(),
            "user": user,
            "query": query,
            "assistant": assistant,
            "context": context,
        }
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def recent(self, n: int = 10) -> List[Dict[str, Any]]:
        items = []
        if os.path.exists(self.path):
            with open(self.path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        items.append(json.loads(line))
        return items[-n:]
    
    def count(self) -> int:
        """Toplam sohbet sayısı"""
        return len(self.recent(1000))


# =========================
# Ollama Model Wrapper (Geliştirilmiş)
# =========================
class Qwen3VLAgent:
    def __init__(self, model_id: str = MODEL_ID, api_url: str = OLLAMA_API_URL, api_key: Optional[str] = None):
        self.logger = Logger()
        
        self.logger.log("Init", "info", "Ollama Agent başlatılıyor...")
        
        # Ollama Client'ı kur
        headers = {}
        if api_key:
            headers['Authorization'] = f'Bearer {api_key}'
        
        self.client = Client(
            host=api_url,
            headers=headers if headers else None
        )
        self.model_id = model_id
        self.logger.log("Init:Client", "success", f"Ollama Client bağlandı: {api_url}")

        self.stm = ShortTermMemory()
        self.ltm = LongTermMemory()
        self.episodic = EpisodicLog()
        
        self.logger.log("Init:Memory", "success", "Bellek sistemleri hazır (STM, LTM, Episodic)")

    def build_prompt_with_memory(self, user_msg: str, retrieved: List[Dict[str, Any]]) -> str:
        preface = "🧠 **UZUN VADELİ BELLEKTEN ANIMSAR:**\n"
        if not retrieved:
            preface += "(Anımsatma yok.)\n"
        else:
            for i, r in enumerate(retrieved, 1):
                preface += f"  [{i}] {r['text']}\n"
        preface += f"\n👤 **SORGUNUZ:** {user_msg}"
        return preface

    def chat(self, user_text: str, top_p=0.9, temperature=0.6) -> Dict[str, Any]:
        """
        Chat fonksiyonu - detaylı logging ile
        Döndürür: { "response": str, "logs": [], "memory_state": {}, "metadata": {} }
        """
        self.logger.clear()
        self.logger.log("Chat:Start", "info", "Yeni sohbet başladı")
        
        # 1) LTM'den geri çağır
        self.logger.log("Chat:Step1", "processing", "Uzun vadeli bellekten relevent bilgiler aranıyor...")
        retrieved = self.ltm.query(user_text, k=4)
        self.logger.log("Chat:Step1", "success", f"{len(retrieved)} ilgili bellek bulundu", 
                       {"retrieved_items": len(retrieved)})

        # 2) Promptu RAG ile birleştir
        self.logger.log("Chat:Step2", "processing", "RAG promptu oluşturuluyor...")
        rag_prompt = self.build_prompt_with_memory(user_text, retrieved)
        self.logger.log("Chat:Step2", "success", "RAG promptu hazır")

        # 3) Kısa vadeli belleğe ekle
        self.logger.log("Chat:Step3", "processing", "Kısa vadeli belleğe sorgu ekleniyor...")
        self.stm.add("user", rag_prompt)
        self.logger.log("Chat:Step3", "success", "STM güncellendi", 
                       {"stm_message_count": len(self.stm.messages)})

        # 4) Ollama API'sine gönder
        self.logger.log("Chat:Step4", "processing", f"API'ye istek gönderiliyor (Model: {self.model_id})...")
        messages = self.stm.as_chat()
        
        text = ""
        error = None
        try:
            response = self.client.chat(
                model=self.model_id,
                messages=messages,
                stream=False,
                options={
                    'temperature': temperature,
                    'top_p': top_p,
                }
            )
            text = response['message']['content']
            self.logger.log("Chat:Step4", "success", "Model yanıt verdi", 
                           {"response_length": len(text)})
        except Exception as e:
            error = str(e)
            text = f"❌ Hata: {error}\nLütfen API key'i kontrol edin."
            self.logger.log("Chat:Step4", "error", f"API Hatası: {error}")

        # 5) Epizodik log + bellek
        self.logger.log("Chat:Step5", "processing", "Sohbet geçmişine kaydediliyor...")
        self.episodic.append(user="user", query=user_text, assistant=text, context={"retrieved": retrieved})
        
        if len(user_text) <= 500:
            self.ltm.upsert([user_text], metadatas=[{"source": "chat", "ts": now_ts()}])
        
        self.logger.log("Chat:Step5", "success", "Bellek güncellendi")

        # 6) STM'e asistan cevabını ekle
        self.stm.add("assistant", text)

        # Metadata ve durum bilgileri
        memory_state = {
            "stm": self.stm.get_summary(),
            "ltm": self.ltm.get_summary(),
            "episodic_count": self.episodic.count()
        }

        self.logger.log("Chat:Complete", "success", "Sohbet tamamlandı")

        return {
            "response": text,
            "logs": self.logger.get_logs(),
            "memory_state": memory_state,
            "metadata": {
                "model": self.model_id,
                "error": error,
                "retrieved_items": len(retrieved)
            }
        }


# =========================
# Demo
# =========================
if __name__ == "__main__":
    if not OLLAMA_API_KEY:
        print("⚠️  OLLAMA_API_KEY environment variable'ı ayarlanmamış!")
        print("Lütfen .env dosyasına ekleyin veya set OLLAMA_API_KEY=your_key komutunu çalıştırın")
    
    print("\n" + "="*60)
    print("🤖 GÜNLÜK KULLLANIM DEMO - Qwen3-VL Bellek Ajanı")
    print("="*60 + "\n")
    
    agent = Qwen3VLAgent(api_key=OLLAMA_API_KEY)

    # Demo sorular
    demo_questions = [
        "Python'da list comprehension nasıl çalışır? Örnek ver.",
        "Daha karmaşık bir örnek ile açıklayabilir misin?",
        "Bu bilgiyi ileriki sohbetlerde hatırlamam lazım."
    ]

    for i, question in enumerate(demo_questions, 1):
        print(f"\n{'='*60}")
        print(f"📌 SORU {i}: {question}")
        print("="*60)
        
        result = agent.chat(question)
        
        # Adımları göster
        print("\n🔍 İŞLEM ADIMLARI:")
        for log in result["logs"]:
            status_icon = {
                "info": "ℹ️",
                "processing": "⚙️",
                "success": "✅",
                "warning": "⚠️",
                "error": "❌"
            }.get(log["status"], "•")
            print(f"  {status_icon} [{log['step']}] {log['message']}")
        
        # Yanıt
        print(f"\n🤖 CEVAP:")
        print(result["response"][:500] + ("..." if len(result["response"]) > 500 else ""))
        
        # Bellek durumu
        print(f"\n💾 BELLEK DURUMU:")
        print(f"  📝 Kısa Vadeli Bellek: {result['memory_state']['stm']['message_count']} mesaj")
        print(f"  🗂️  Uzun Vadeli Bellek: {result['memory_state']['ltm']['document_count']} doküman")
        print(f"  📊 Toplam Sohbet: {result['memory_state']['episodic_count']}")
        
        time.sleep(1)
