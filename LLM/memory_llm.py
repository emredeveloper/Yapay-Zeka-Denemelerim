import os
import json
import time
import base64
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path

from ollama import Client

# .env dosyasından environment variable'ları yükle
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

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
VISUAL_CACHE_DIR = "./visual_cache"


# =========================
# Yardımcılar
# =========================
def ensure_jsonl(file_path: str):
    if not os.path.exists(file_path):
        with open(file_path, "w", encoding="utf-8"):
            pass

def ensure_dir(dir_path: str):
    os.makedirs(dir_path, exist_ok=True)

def now_ts():
    return time.strftime("%Y-%m-%d %H:%M:%S")

def image_to_base64(image_path: str) -> str:
    """Resmi base64'e çevir"""
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

def save_visual(base64_data: str, filename: str) -> str:
    """Base64 resmi kaydet"""
    ensure_dir(VISUAL_CACHE_DIR)
    filepath = os.path.join(VISUAL_CACHE_DIR, filename)
    with open(filepath, "wb") as f:
        f.write(base64.b64decode(base64_data))
    return filepath


# =========================
# Kısa Vadeli Bellek
# =========================
@dataclass
class ShortTermMemory:
    token_limit: int = SHORT_TERM_TOKEN_LIMIT
    messages: List[Dict[str, Any]] = field(default_factory=list)

    def add(self, role: str, text: str, image_base64: Optional[str] = None):
        content = [{"type": "text", "text": text}]
        if image_base64:
            content.insert(0, {"type": "image", "image": image_base64})
        self.messages.append({"role": role, "content": content})
        while len(str(self.messages)) > self.token_limit * 4 and len(self.messages) > 1:
            self.messages.pop(0)

    def as_chat(self) -> List[Dict[str, Any]]:
        result = []
        for msg in self.messages:
            role = msg["role"]
            text_content = ""
            images = []
            
            # Content'ten text ve image'ları ayır
            if isinstance(msg["content"], list):
                for item in msg["content"]:
                    if isinstance(item, dict):
                        if item.get("type") == "text":
                            text_content += item["text"]
                        elif item.get("type") == "image":
                            images.append(item["image"])
            
            # Message oluştur
            msg_obj = {"role": role, "content": text_content or ""}
            
            # Eğer görsel varsa images field'ına ekle
            if images:
                msg_obj["images"] = images
            
            result.append(msg_obj)
        
        return result
    
    def get_summary(self) -> Dict[str, Any]:
        """Bellek özeti"""
        image_count = sum(
            1 for msg in self.messages 
            for item in (msg.get("content") or []) 
            if isinstance(item, dict) and item.get("type") == "image"
        )
        return {
            "message_count": len(self.messages),
            "image_count": image_count,
            "size_estimate": len(str(self.messages)),
            "limit": self.token_limit * 4,
            "messages": self.as_chat()
        }


# =========================
# Uzun Vadeli Bellek
# =========================
class LongTermMemory:
    def __init__(self, coll_name: str = COLL_NAME, persist_dir: str = CHROMA_DIR):
        ensure_dir(persist_dir)
        self.persist_dir = persist_dir
        self.client = chromadb.PersistentClient(path=persist_dir, settings=Settings(allow_reset=False))
        try:
            self.coll = self.client.get_collection(coll_name)
        except Exception:
            self.coll = self.client.create_collection(coll_name)
        self.embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.logger = Logger()
        self.visual_store = {}
        self.visual_store_path = os.path.join(persist_dir, "visuals.json")
        self.load_visuals()

    def load_visuals(self):
        """Disk'ten görselleri yükle"""
        if os.path.exists(self.visual_store_path):
            try:
                with open(self.visual_store_path, 'r', encoding='utf-8') as f:
                    self.visual_store = json.load(f)
                    self.logger.log("LTM:Load", "success", f"{len(self.visual_store)} görsel yüklendi")
            except Exception as e:
                self.logger.log("LTM:Load", "error", f"Görsel yükleme hatası: {e}")

    def save_visuals(self):
        """Görselleri disk'e kaydet"""
        try:
            with open(self.visual_store_path, 'w', encoding='utf-8') as f:
                json.dump(self.visual_store, f, ensure_ascii=False)
        except Exception as e:
            self.logger.log("LTM:Save", "error", f"Görsel kayıt hatası: {e}")

    def upsert(self, docs: List[str], metadatas: Optional[List[Dict[str, Any]]] = None, 
               ids: Optional[List[str]] = None, visuals: Optional[List[Dict[str, str]]] = None):
        embeddings = self.embedder.encode(docs, normalize_embeddings=True).tolist()
        if ids is None:
            ids = [f"doc-{int(time.time()*1e6)}-{i}" for i in range(len(docs))]
        
        if metadatas is None:
            metadatas = [{} for _ in docs]
        
        if visuals:
            for i, visual in enumerate(visuals):
                if visual and i < len(metadatas):
                    metadatas[i]["has_image"] = True
                    self.visual_store[ids[i]] = visual
        
        self.coll.upsert(ids=ids, documents=docs, metadatas=metadatas, embeddings=embeddings)
        self.save_visuals()  # Disk'e kaydet
        visual_count = len([v for v in (visuals or []) if v])
        self.logger.log("LTM:Upsert", "success", f"{len(docs)} doküman (görsel: {visual_count}) belleğe eklendi")

    def query(self, q: str, k: int = 5) -> List[Dict[str, Any]]:
        q_emb = self.embedder.encode([q], normalize_embeddings=True).tolist()
        res = self.coll.query(query_embeddings=q_emb, n_results=k)
        out = []
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        ids = res.get("ids", [[]])[0]
        
        for doc_id, doc, meta in zip(ids, docs, metas):
            item = {"text": doc, "meta": meta, "id": doc_id}
            if self.visual_store.get(doc_id):
                item["visual"] = self.visual_store[doc_id]
            out.append(item)
        return out
    
    def count(self) -> int:
        return self.coll.count()
    
    def get_visual_count(self) -> int:
        return len(self.visual_store)
    
    def get_all_visuals(self) -> List[Dict[str, Any]]:
        """Tüm görselleri getir"""
        return [
            {"id": vid, **vdata} 
            for vid, vdata in self.visual_store.items()
        ]
    
    def get_summary(self) -> Dict[str, Any]:
        return {
            "document_count": self.count(),
            "visual_count": self.get_visual_count(),
            "embedder": "all-MiniLM-L6-v2"
        }


# =========================
# Epizodik Log
# =========================
class EpisodicLog:
    def __init__(self, path: str = EPISODIC_LOG):
        self.path = path
        ensure_jsonl(self.path)

    def append(self, user: str, query: str, assistant: str, context: Dict[str, Any], 
               image_path: Optional[str] = None, visual_id: Optional[str] = None):
        rec = {
            "ts": now_ts(),
            "user": user,
            "query": query,
            "assistant": assistant,
            "context": context,
            "has_image": bool(image_path),
            "visual_id": visual_id
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
        return len(self.recent(1000))
    
    def get_stats(self) -> Dict[str, Any]:
        items = self.recent(1000)
        return {
            "total": len(items),
            "with_images": sum(1 for item in items if item.get("has_image")),
            "avg_query_length": sum(len(item.get("query", "")) for item in items) // len(items) if items else 0,
            "avg_response_length": sum(len(item.get("assistant", "")) for item in items) // len(items) if items else 0,
        }


# =========================
# Ollama Model Wrapper
# =========================
class Qwen3VLAgent:
    def __init__(self, model_id: str = MODEL_ID, api_url: str = OLLAMA_API_URL, api_key: Optional[str] = None):
        self.logger = Logger()
        self.logger.log("Init", "info", "Ollama Agent başlatılıyor...")
        
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
                visual_note = " [GÖRSEL MEVCUL]" if r.get("visual") else ""
                preface += f"  [{i}] {r['text']}{visual_note}\n"
        preface += f"\n👤 **SORGUNUZ:** {user_msg}"
        return preface

    def chat(self, user_text: str, image_base64: Optional[str] = None, 
             top_p=0.9, temperature=0.6) -> Dict[str, Any]:
        self.logger.clear()
        self.logger.log("Chat:Start", "info", "Yeni sohbet başladı")
        
        # 1) LTM'den geri çağır
        self.logger.log("Chat:Step1", "processing", "Uzun vadeli bellekten relevent bilgiler aranıyor...")
        retrieved = self.ltm.query(user_text, k=4)
        self.logger.log("Chat:Step1", "success", f"{len(retrieved)} ilgili bellek bulundu")

        # 2) RAG promptu
        self.logger.log("Chat:Step2", "processing", "RAG promptu oluşturuluyor...")
        rag_prompt = self.build_prompt_with_memory(user_text, retrieved)
        if image_base64:
            rag_prompt += "\n\n🖼️ [GÖRSEL ANALIZ YAPILACAK]"
        self.logger.log("Chat:Step2", "success", "RAG promptu hazır")

        # 3) STM'ye ekle
        self.logger.log("Chat:Step3", "processing", "Belleğe ekleniyor...")
        self.stm.add("user", rag_prompt, image_base64=image_base64)
        self.logger.log("Chat:Step3", "success", f"STM güncellendi (görsel: {'Var' if image_base64 else 'Yok'})")

        # 4) API'ye gönder
        self.logger.log("Chat:Step4", "processing", f"API'ye istek gönderiliyor...")
        messages = self.stm.as_chat()
        
        text = ""
        error = None
        try:
            response = self.client.chat(
                model=self.model_id,
                messages=messages,
                stream=False,
                options={'temperature': temperature, 'top_p': top_p}
            )
            text = response['message']['content']
            self.logger.log("Chat:Step4", "success", "Model yanıt verdi")
        except Exception as e:
            error = str(e)
            text = f"❌ Hata: {error}"
            self.logger.log("Chat:Step4", "error", f"API Hatası: {error}")

        # 5) Bellek güncelle
        self.logger.log("Chat:Step5", "processing", "Bellek güncelleniyor...")
        
        # Visual ID oluştur (varsa)
        visual_id = None
        if image_base64:
            visual_id = f"vis-{int(time.time()*1e6)}"
        
        self.episodic.append(user="user", query=user_text, assistant=text, 
                            context={"retrieved": retrieved}, image_path=image_base64, visual_id=visual_id)
        
        # Görsel varsa LTM'ye ekle
        visuals = [{"base64": image_base64, "description": user_text}] if image_base64 else None
        if len(user_text) <= 500:
            ids = [f"doc-{int(time.time()*1e6)}-0"]
            self.ltm.upsert([user_text], metadatas=[{"source": "chat", "ts": now_ts(), "visual_id": visual_id}], 
                           ids=ids, visuals=visuals)
        
        self.logger.log("Chat:Step5", "success", "Bellek güncellendi")

        self.stm.add("assistant", text)

        memory_state = {
            "stm": self.stm.get_summary(),
            "ltm": self.ltm.get_summary(),
            "episodic_count": self.episodic.count(),
            "episodic_stats": self.episodic.get_stats()
        }

        self.logger.log("Chat:Complete", "success", "Sohbet tamamlandı")

        return {
            "response": text,
            "logs": self.logger.get_logs(),
            "memory_state": memory_state,
            "prompt_used": rag_prompt,
            "metadata": {
                "model": self.model_id,
                "error": error,
                "retrieved_items": len(retrieved),
                "has_image": bool(image_base64)
            }
        }


if __name__ == "__main__":
    if not OLLAMA_API_KEY:
        print("⚠️  OLLAMA_API_KEY environment variable'ı ayarlanmamış!")
    
    print("\n" + "="*60)
    print("🤖 GÜNLÜK KULLLANIM DEMO - Qwen3-VL Bellek Ajanı")
    print("="*60 + "\n")
    
    agent = Qwen3VLAgent(api_key=OLLAMA_API_KEY)

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
        
        print("\n🔍 İŞLEM ADIMLARI:")
        for log in result["logs"]:
            status_icon = {"info": "ℹ️", "processing": "⚙️", "success": "✅", "warning": "⚠️", "error": "❌"}.get(log["status"], "•")
            print(f"  {status_icon} [{log['step']}] {log['message']}")
        
        print(f"\n🤖 CEVAP:")
        print(result["response"][:500] + ("..." if len(result["response"]) > 500 else ""))
        
        print(f"\n💾 BELLEK DURUMU:")
        print(f"  📝 Kısa Vadeli: {result['memory_state']['stm']['message_count']} mesaj, {result['memory_state']['stm']['image_count']} görsel")
        print(f"  🗂️  Uzun Vadeli: {result['memory_state']['ltm']['document_count']} doküman, {result['memory_state']['ltm']['visual_count']} görsel")
        
        time.sleep(1)
