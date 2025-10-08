"""
Ollama API entegrasyonu
- granite3.1:2b: Metin tabanlı Q&A ve analiz
- qwen2.5-vl:3b: Görsel analiz ve açıklama
"""

import requests
import json
import base64
from pathlib import Path
from typing import List, Dict, Optional, Union


class OllamaClient:
    """Ollama API istemcisi"""
    
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        self.text_model = "granite4:tiny-h"  # Kullanıcının önerdiği model
        self.vision_model = "qwen2.5vl:3b"   # Mevcut vision model
    
    def check_connection(self) -> bool:
        """Ollama bağlantısını kontrol et"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def list_models(self) -> List[str]:
        """Yüklü modelleri listele"""
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            if response.status_code == 200:
                data = response.json()
                return [model['name'] for model in data.get('models', [])]
            return []
        except:
            return []
    
    def _encode_image(self, image_path: str) -> str:
        """Görseli base64'e çevir"""
        with open(image_path, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')
    
    def generate_text(self, 
                     prompt: str, 
                     system: Optional[str] = None,
                     context: Optional[str] = None,
                     stream: bool = False) -> Union[str, Dict]:
        """Metin üretimi (granite3.1:2b)"""
        try:
            # Prompt oluştur
            full_prompt = prompt
            if context:
                full_prompt = f"Context:\n{context}\n\nQuestion: {prompt}"
            
            payload = {
                "model": self.text_model,
                "prompt": full_prompt,
                "stream": stream,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "top_k": 40,
                }
            }
            
            if system:
                payload["system"] = system
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                if stream:
                    return response
                else:
                    result = response.json()
                    return result.get('response', '')
            else:
                return f"Hata: {response.status_code}"
                
        except Exception as e:
            return f"Hata: {str(e)}"
    
    def analyze_image(self, 
                     image_path: str, 
                     question: str = "Bu görselde ne görüyorsun? Detaylı açıkla.",
                     stream: bool = False) -> Union[str, Dict]:
        """Görsel analizi (qwen2.5-vl:3b)"""
        try:
            # Görseli encode et
            image_b64 = self._encode_image(image_path)
            
            payload = {
                "model": self.vision_model,
                "prompt": question,
                "images": [image_b64],
                "stream": stream,
                "options": {
                    "temperature": 0.5,
                    "top_p": 0.9,
                }
            }
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                if stream:
                    return response
                else:
                    result = response.json()
                    return result.get('response', '')
            else:
                return f"Hata: {response.status_code}"
                
        except Exception as e:
            return f"Hata: {str(e)}"
    
    def analyze_multiple_images(self,
                               image_paths: List[str],
                               question: str) -> List[Dict[str, str]]:
        """Çoklu görsel analizi"""
        results = []
        for img_path in image_paths:
            analysis = self.analyze_image(img_path, question)
            results.append({
                'image': img_path,
                'analysis': analysis
            })
        return results
    
    def answer_question_with_context(self,
                                    question: str,
                                    transcript: str,
                                    video_info: Dict,
                                    relevant_frames: Optional[List[str]] = None) -> str:
        """Video bağlamında soru cevaplama"""
        
        # System prompt - daha iyi Türkçe
        system = """Sen bir video analiz asistanısın. Kullanıcıya verilen video transkripti ve 
        video bilgilerine dayanarak sorularını yanıtlıyorsun. 
        
        Kurallar:
        1. Sadece verilen transkript ve bilgileri kullan
        2. Net, anlaşılır ve Türkçe cevap ver
        3. Eğer bilgi transkriptte yoksa, açıkça söyle
        4. Mümkünse zaman damgalarını (örn: "5. dakikada...") belirt
        5. Kısa ve öz cevaplar ver, gereksiz detaya girme"""
        
        # Transcript'i temizle ve sınırla
        clean_transcript = transcript.strip()
        if len(clean_transcript) > 8000:
            clean_transcript = clean_transcript[:8000] + "... (transkript devam ediyor)"
        
        # Context hazırla - daha yapılandırılmış
        context = f"""=== VIDEO BİLGİLERİ ===
Başlık: {video_info.get('title', 'Bilinmiyor')}
Kanal: {video_info.get('channel', 'Bilinmiyor')}
Süre: {video_info.get('duration', 'Bilinmiyor')}
Görüntülenme: {video_info.get('views', 'Bilinmiyor')}

=== VİDEO TRANSKRİPTİ ===
{clean_transcript}
"""
        
        # Eğer görseller varsa, onları da analiz et
        if relevant_frames and len(relevant_frames) > 0:
            context += "\n\n=== GÖRSEL ANALİZLER ===\n"
            for i, frame_path in enumerate(relevant_frames[:2], 1):  # İlk 2 frame
                try:
                    img_analysis = self.analyze_image(
                        frame_path, 
                        "Bu video karesi neyi gösteriyor? Çok kısa açıkla (max 2 cümle)."
                    )
                    context += f"Görsel {i}: {img_analysis}\n"
                except Exception as e:
                    context += f"Görsel {i}: Analiz edilemedi\n"
        
        return self.generate_text(question, system=system, context=context)
    
    def semantic_search(self,
                       query: str,
                       transcript_segments: List[Dict],
                       top_k: int = 5) -> List[Dict]:
        """Semantik arama (basit versiyon)"""
        # LLM kullanarak alakalı segmentleri bul
        segments_text = "\n".join([
            f"[{i}] {seg['text']} (Zaman: {seg['start']:.1f}s)"
            for i, seg in enumerate(transcript_segments[:100])  # İlk 100 segment
        ])
        
        prompt = f"""
Aşağıdaki transkript segmentlerinden "{query}" ile en alakalı {top_k} tanesini bul.
Sadece segment numaralarını virgülle ayırarak yaz (örnek: 1,5,12,20,45).

Segmentler:
{segments_text}

En alakalı {top_k} segment numarası:"""
        
        try:
            response = self.generate_text(prompt)
            # Numaraları çıkar
            numbers = [int(n.strip()) for n in response.split(',') if n.strip().isdigit()]
            
            results = []
            for idx in numbers[:top_k]:
                if idx < len(transcript_segments):
                    results.append(transcript_segments[idx])
            
            return results
        except:
            # Basit text matching fallback
            results = []
            query_lower = query.lower()
            for seg in transcript_segments:
                if query_lower in seg['text'].lower():
                    results.append(seg)
                    if len(results) >= top_k:
                        break
            return results


# Test fonksiyonu
def test_ollama():
    """Ollama bağlantısını test et"""
    client = OllamaClient()
    
    print("🔍 Ollama Bağlantı Testi")
    print("="*50)
    
    if client.check_connection():
        print("✅ Ollama'ya bağlantı başarılı!")
        
        models = client.list_models()
        print(f"\n📦 Yüklü Modeller ({len(models)}):")
        for model in models:
            print(f"  - {model}")
        
        # Model kontrolü
        if client.text_model in models:
            print(f"\n✅ Metin modeli mevcut: {client.text_model}")
        else:
            print(f"\n⚠️  Metin modeli eksik: {client.text_model}")
            print(f"   İndirmek için: ollama pull {client.text_model}")
        
        if client.vision_model in models:
            print(f"✅ Görsel modeli mevcut: {client.vision_model}")
        else:
            print(f"⚠️  Görsel modeli eksik: {client.vision_model}")
            print(f"   İndirmek için: ollama pull {client.vision_model}")
        
        # Basit test
        if client.text_model in models:
            print("\n🧪 Test Sorusu...")
            response = client.generate_text("Merhaba! Nasılsın?")
            print(f"📝 Cevap: {response[:100]}...")
        
    else:
        print("❌ Ollama'ya bağlanılamadı!")
        print("   Ollama çalışıyor mu? Kontrol et: ollama list")
    
    print("="*50)


if __name__ == "__main__":
    test_ollama()
