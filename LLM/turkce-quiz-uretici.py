"""
Türkçe Quiz Üretici - Structured Output Örneği
Web scraping ile makale/konu içeriği çekip quiz soruları üretir.
"""
from google import genai
from pydantic import BaseModel, Field
from typing import List, Literal, Optional
import json
import requests
from bs4 import BeautifulSoup
import re

client = genai.Client(api_key="AIzaSyBDTJmH-oCCq9Td7G6g93_93yHH3gTcJkg")

class Secenek(BaseModel):
    """Soru seçeneği"""
    metin: str = Field(description="Seçenek metni")
    dogru: bool = Field(description="Doğru cevap mı?")

class Soru(BaseModel):
    """Quiz sorusu"""
    soru: str = Field(description="Soru metni")
    zorluk: Literal["kolay", "orta", "zor"] = Field(description="Soru zorluğu")
    tip: Literal["çoktan_seçmeli", "doğru_yanlış", "boşluk_doldurma"] = Field(description="Soru tipi")
    secenekler: List[Secenek] = Field(description="Seçenekler (çoktan seçmeli için)")
    dogru_cevap: str = Field(description="Doğru cevap açıklaması")
    aciklama: str = Field(description="Cevap açıklaması")

class Quiz(BaseModel):
    """Quiz seti"""
    konu: str = Field(description="Quiz konusu")
    baslik: str = Field(description="Quiz başlığı")
    aciklama: str = Field(description="Quiz açıklaması")
    sorular: List[Soru] = Field(description="Quiz soruları", min_items=5, max_items=10)
    toplam_puan: int = Field(description="Toplam puan")

def makale_cek(url: str) -> Optional[str]:
    """
    Web sitesinden makale/konu içeriğini çeker.
    
    Args:
        url: Makale URL'i
        
    Returns:
        str: Makale metni veya None
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.encoding = 'utf-8'
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Article veya main content alanını bul
        article = (soup.find('article') or 
                  soup.find('main') or 
                  soup.find('div', class_=re.compile(r'article|content|post|makale', re.I)))
        
        if article:
            # Script ve style etiketlerini kaldır
            for script in article(["script", "style", "nav", "header", "footer", "aside"]):
                script.decompose()
            
            # Metni al ve temizle
            text = article.get_text(separator=' ', strip=True)
            text = re.sub(r'\s+', ' ', text)
            return text[:8000]  # İlk 8000 karakteri al
        
        # Alternatif: p etiketlerinden metin topla
        paragraphs = soup.find_all('p')
        if paragraphs:
            text = ' '.join([p.get_text(strip=True) for p in paragraphs])
            return text[:8000]
        
        return None
        
    except Exception as e:
        print(f"❌ Hata: {e}")
        return None

def quiz_uret(konu: str, soru_sayisi: int = 5, zorluk: str = "orta", icerik_metni: Optional[str] = None) -> Quiz:
    """
    Belirli bir konudan quiz soruları üretir.
    
    Args:
        konu: Quiz konusu
        soru_sayisi: Soru sayısı (5-10 arası)
        zorluk: Zorluk seviyesi (kolay, orta, zor)
        
    Returns:
        Quiz: Üretilen quiz
    """
    schema = Quiz.model_json_schema()
    
    # İçerik metni varsa onu kullan, yoksa sadece konu adını kullan
    if icerik_metni:
        prompt = f"""Aşağıdaki makale içeriğine göre Türkçe quiz soruları üret.

MAKALE İÇERİĞİ:
{icerik_metni}

KONU: {konu}
SORU SAYISI: {soru_sayisi}
ZORLUK: {zorluk}

Lütfen:
- Quiz başlığı ve açıklaması oluştur
- Çeşitli soru tipleri kullan (çoktan seçmeli, doğru/yanlış, boşluk doldurma)
- Her soru için doğru cevap ve açıklama ekle
- Soruları belirtilen zorluk seviyesinde hazırla
- Toplam puanı hesapla (her soru 10 puan)

Tüm çıktılar Türkçe olmalıdır."""
    else:
        prompt = f"""Aşağıdaki konu hakkında Türkçe quiz soruları üret.

KONU: {konu}
SORU SAYISI: {soru_sayisi}
ZORLUK: {zorluk}

Lütfen:
- Quiz başlığı ve açıklaması oluştur
- Çeşitli soru tipleri kullan (çoktan seçmeli, doğru/yanlış, boşluk doldurma)
- Her soru için doğru cevap ve açıklama ekle
- Soruları belirtilen zorluk seviyesinde hazırla
- Toplam puanı hesapla (her soru 10 puan)

Tüm çıktılar Türkçe olmalıdır."""
    
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config={
            "response_mime_type": "application/json",
            "response_json_schema": schema,
        },
    )
    
    json_data = json.loads(response.text)
    return Quiz(**json_data)

# Örnek kullanım
if __name__ == "__main__":
    import sys
    
    print("📚 Quiz Üretici - Türkçe Structured Output Örneği\n")
    
    icerik_metni = None
    
    if len(sys.argv) > 1:
        # Komut satırından URL al
        url = sys.argv[1]
        print(f"🌐 Makale URL'den çekiliyor: {url}\n")
        icerik_metni = makale_cek(url)
        
        if not icerik_metni:
            print("❌ Makale içeriği çekilemedi. Lütfen geçerli bir URL girin.")
            sys.exit(1)
        
        # İçerikten konu çıkarmaya çalış (başlıktan)
        konu = "Makale İçeriği"
        print(f"✅ Makale içeriği çekildi ({len(icerik_metni)} karakter)\n")
    else:
        konu = "Python Programlama Dili Temel Kavramlar"
        print("💡 İpucu: URL ile kullanmak için: python turkce-quiz-uretici.py <MAKALE_URL>\n")
    
    print(f"Konu: {konu}\n")
    print("⏳ Quiz oluşturuluyor...\n")
    
    quiz = quiz_uret(konu, soru_sayisi=5, zorluk="orta", icerik_metni=icerik_metni)
    
    print("=" * 70)
    print(f"📝 {quiz.baslik.upper()}")
    print("=" * 70)
    print(f"\n📖 Açıklama: {quiz.aciklama}")
    print(f"📊 Toplam Puan: {quiz.toplam_puan}\n")
    
    for i, soru in enumerate(quiz.sorular, 1):
        print(f"\n{'─'*70}")
        print(f"SORU {i} ({soru.zorluk.upper()} - {soru.tip.replace('_', ' ').upper()})")
        print(f"{'─'*70}")
        print(f"{soru.soru}\n")
        
        if soru.secenekler:
            for j, secenek in enumerate(soru.secenekler, 1):
                isaret = "✓" if secenek.dogru else " "
                print(f"  {isaret} {j}. {secenek.metin}")
        
        print(f"\n💡 Doğru Cevap: {soru.dogru_cevap}")
        print(f"📚 Açıklama: {soru.aciklama}")
    
    print(f"\n{'='*70}")
    print("✅ Quiz tamamlandı!")

