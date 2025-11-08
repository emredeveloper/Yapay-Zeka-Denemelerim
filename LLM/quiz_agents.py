"""
Multi-Agent Quiz Sistemi - Devrimsel Öğrenme Platformu
Her agent farklı bir görevi üstlenir ve birlikte çalışır.
"""
from google import genai
from pydantic import BaseModel, Field
from typing import List, Literal, Optional, Dict
import json
from datetime import datetime
from collections import defaultdict

client = genai.Client(api_key="AIzaSyBDTJmH-oCCq9Td7G6g93_93yHH3gTcJkg")

# ==================== AGENT 1: QUIZ GENERATOR AGENT ====================
class QuizGeneratorAgent:
    """Quiz soruları üreten agent"""
    
    def __init__(self):
        self.model = "gemini-2.5-flash"
    
    def generate_quiz(self, konu: str, soru_sayisi: int, zorluk: str, 
                     icerik_metni: Optional[str] = None, 
                     eksik_konular: Optional[List[str]] = None) -> Dict:
        """Quiz üretir, eksik konulara odaklanabilir"""
        
        schema = {
            "type": "object",
            "properties": {
                "baslik": {"type": "string"},
                "aciklama": {"type": "string"},
                "sorular": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "soru": {"type": "string"},
                            "zorluk": {"type": "string", "enum": ["kolay", "orta", "zor"]},
                            "tip": {"type": "string"},
                            "secenekler": {"type": "array"},
                            "dogru_cevap": {"type": "string"},
                            "aciklama": {"type": "string"},
                            "konu": {"type": "string"}
                        }
                    },
                    "minItems": soru_sayisi,
                    "maxItems": soru_sayisi
                }
            }
        }
        
        focus_text = ""
        if eksik_konular:
            focus_text = f"\nÖNEMLİ: Aşağıdaki konularda öğrencinin zorlandığını tespit ettik. Bu konulara özel sorular üret:\n{', '.join(eksik_konular)}"
        
        prompt = f"""Quiz Generator Agent olarak görev yapıyorsun. Aşağıdaki gereksinimlere göre quiz soruları üret.

KONU: {konu}
SORU SAYISI: {soru_sayisi}
ZORLUK: {zorluk}
{focus_text}

Lütfen:
- Çeşitli soru tipleri kullan (çoktan seçmeli, doğru/yanlış, boşluk doldurma)
- Her soru için konu etiketi ekle
- Soruları belirtilen zorluk seviyesinde hazırla
- Tüm çıktılar Türkçe olmalıdır."""
        
        if icerik_metni:
            prompt += f"\n\nMAKALE İÇERİĞİ:\n{icerik_metni[:8000]}"
        
        response = client.models.generate_content(
            model=self.model,
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_json_schema": schema,
            },
        )
        
        return json.loads(response.text)


# ==================== AGENT 2: LEARNING ANALYST AGENT ====================
class LearningAnalystAgent:
    """Öğrenci performansını analiz eden agent"""
    
    def __init__(self):
        self.model = "gemini-2.5-flash"
    
    def analyze_performance(self, quiz_results: Dict, user_answers: Dict) -> Dict:
        """Öğrenci performansını analiz eder"""
        
        schema = {
            "type": "object",
            "properties": {
                "genel_puan": {"type": "number"},
                "yuzde": {"type": "number"},
                "eksik_konular": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "guclu_konular": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "ozer_oncesi": {"type": "string"},
                "ozer_sonrasi": {"type": "string"},
                "ogrenme_yolu": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "tahmini_sure": {"type": "string"}
            }
        }
        
        # Soru bazında analiz
        soru_analiz = []
        for idx, soru in enumerate(quiz_results['sorular']):
            user_answer = user_answers.get(idx, None)
            is_correct = False
            if user_answer is not None and soru.get('secenekler'):
                correct_idx = next((i for i, s in enumerate(soru['secenekler']) if s.get('dogru')), None)
                is_correct = user_answer == correct_idx
            
            soru_analiz.append({
                "soru": soru.get('soru', ''),
                "konu": soru.get('konu', ''),
                "dogru": is_correct,
                "zorluk": soru.get('zorluk', 'orta')
            })
        
        analiz_metni = json.dumps(soru_analiz, ensure_ascii=False)
        
        prompt = f"""Learning Analyst Agent olarak görev yapıyorsun. Öğrenci performansını analiz et ve detaylı rapor hazırla.

ÖĞRENCI CEVAPLARI VE SONUÇLAR:
{analiz_metni}

Lütfen:
- Eksik konuları tespit et
- Güçlü konuları belirle
- Öğrenme yolunu öner
- Tahmini iyileştirme süresini belirt
- Kişiselleştirilmiş öneriler sun
- Tüm çıktılar Türkçe olmalıdır."""
        
        response = client.models.generate_content(
            model=self.model,
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_json_schema": schema,
            },
        )
        
        return json.loads(response.text)


# ==================== AGENT 3: ADAPTIVE AGENT ====================
class AdaptiveAgent:
    """Zorluk seviyesini ve soru tipini adapte eden agent"""
    
    def __init__(self):
        self.learning_history = defaultdict(list)
    
    def adapt_difficulty(self, user_id: str, performance: Dict, 
                        current_difficulty: str) -> str:
        """Performansa göre zorluk seviyesini ayarlar"""
        
        self.learning_history[user_id].append({
            'performance': performance,
            'difficulty': current_difficulty,
            'timestamp': datetime.now().isoformat()
        })
        
        yuzde = performance.get('yuzde', 50)
        
        if yuzde >= 90:
            # Çok iyi performans, zorluğu artır
            if current_difficulty == 'kolay':
                return 'orta'
            elif current_difficulty == 'orta':
                return 'zor'
            return 'zor'
        elif yuzde >= 70:
            # İyi performans, aynı seviyede kal
            return current_difficulty
        elif yuzde >= 50:
            # Orta performans, biraz düşür
            if current_difficulty == 'zor':
                return 'orta'
            return current_difficulty
        else:
            # Düşük performans, seviyeyi düşür
            if current_difficulty == 'zor':
                return 'orta'
            elif current_difficulty == 'orta':
                return 'kolay'
            return 'kolay'
    
    def suggest_next_quiz(self, eksik_konular: List[str], 
                         soru_sayisi: int = 5) -> Dict:
        """Eksik konulara göre sonraki quiz'i önerir"""
        return {
            'konu': eksik_konular[0] if eksik_konular else 'Genel Tekrar',
            'soru_sayisi': soru_sayisi,
            'zorluk': 'orta',
            'odak_konular': eksik_konular[:3]
        }


# ==================== AGENT 4: TUTOR AGENT ====================
class TutorAgent:
    """Anlık yardım ve açıklama sağlayan agent"""
    
    def __init__(self):
        self.model = "gemini-2.5-flash"
    
    def get_hint(self, soru: Dict, user_attempt: int = 1) -> str:
        """Soru için ipucu verir"""
        
        schema = {
            "type": "object",
            "properties": {
                "ipucu": {"type": "string"},
                "aciklama": {"type": "string"}
            }
        }
        
        attempt_text = "ilk" if user_attempt == 1 else "ikinci" if user_attempt == 2 else "son"
        
        prompt = f"""Tutor Agent olarak görev yapıyorsun. Öğrenciye yardımcı ol ama cevabı direkt verme.

SORU:
{soru.get('soru', '')}

SEÇENEKLER:
{json.dumps(soru.get('secenekler', []), ensure_ascii=False)}

Öğrenci {attempt_text} denemesinde. Uygun bir ipucu ver ve kavramı açıkla. Tüm çıktılar Türkçe olmalıdır."""
        
        response = client.models.generate_content(
            model=self.model,
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_json_schema": schema,
            },
        )
        
        result = json.loads(response.text)
        return result.get('ipucu', '') + "\n\n" + result.get('aciklama', '')
    
    def explain_concept(self, konu: str) -> str:
        """Konuyu detaylı açıklar"""
        
        prompt = f"""Tutor Agent olarak görev yapyorsun. Aşağıdaki konuyu öğrenciye basit ve anlaşılır şekilde açıkla.

KONU: {konu}

Lütfen:
- Örneklerle açıkla
- Pratik ipuçları ver
- Tüm çıktılar Türkçe olmalıdır."""
        
        response = client.models.generate_content(
            model=self.model,
            contents=prompt
        )
        
        return response.text


# ==================== AGENT 5: GAMIFICATION AGENT ====================
class GamificationAgent:
    """Badge, level, achievement sistemi yöneten agent"""
    
    def __init__(self):
        self.achievements = {
            'ilk_quiz': {'name': 'İlk Adım', 'desc': 'İlk quizini tamamladın!'},
            'mukemmel': {'name': 'Mükemmel!', 'desc': '100 puan aldın!'},
            'seri': {'name': 'Seri', 'desc': '5 quiz üst üste tamamladın!'},
            'uzman': {'name': 'Uzman', 'desc': 'Zor seviyede 10 quiz tamamladın!'},
            'ogrenci': {'name': 'Öğrenci', 'desc': '50 quiz tamamladın!'},
            'hazir': {'name': 'Hazır', 'desc': '10 quiz tamamladın!'},
        }
    
    def calculate_level(self, total_quizzes: int, total_score: int) -> Dict:
        """Kullanıcı seviyesini hesaplar"""
        level = min(1 + (total_quizzes // 5), 20)  # Max level 20
        exp = total_score
        next_level_exp = level * 100
        
        return {
            'level': level,
            'exp': exp,
            'next_level_exp': next_level_exp,
            'progress': (exp % 100) / 100 if exp > 0 else 0
        }
    
    def check_achievements(self, stats: Dict) -> List[str]:
        """Kazanılan başarımları kontrol eder"""
        earned = []
        
        if stats.get('total_quizzes', 0) >= 1:
            earned.append('ilk_quiz')
        if stats.get('total_quizzes', 0) >= 10:
            earned.append('hazir')
        if stats.get('total_quizzes', 0) >= 50:
            earned.append('ogrenci')
        if stats.get('perfect_scores', 0) >= 1:
            earned.append('mukemmel')
        if stats.get('streak', 0) >= 5:
            earned.append('seri')
        if stats.get('hard_quizzes', 0) >= 10:
            earned.append('uzman')
        
        return earned
    
    def get_achievement_info(self, achievement_id: str) -> Dict:
        """Başarım bilgilerini döndürür"""
        return self.achievements.get(achievement_id, {})

