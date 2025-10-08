# 🎥 YouTube Video Analiz Aracı + 🤖 AI

<div align="center">

```
╔══════════════════════════════════════════════════════════════════╗
║  🚀 AI Destekli YouTube Video Analizi                            ║
║  💬 Soru-Cevap | 🔍 Akıllı Arama | 🖼️ Görsel Analiz             ║
║  ⚡ 70% Daha Hızlı | 🔒 %100 Local & Güvenli                    ║
╚══════════════════════════════════════════════════════════════════╝
```

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0+-green.svg)](https://flask.palletsprojects.com/)
[![Ollama](https://img.shields.io/badge/Ollama-Local%20AI-orange.svg)](https://ollama.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

---

> **🆕 Yeni!** Artık AI destekli video analizi, akıllı arama ve soru-cevap özelliği ile!

YouTube videolarından transkript, görsel frame'ler ve istatistiksel bilgiler çıkaran **+ Local AI entegrasyonlu** kapsamlı Python aracı.

## ⚡ Hızlı Başlangıç

```bash
# 1. Repo'yu klonla
git clone <repo-url>
cd Video-Analysis

# 2. Python paketlerini yükle
pip install -r requirements.txt

# 3. Ollama'yı kur ve modelleri indir
ollama pull granite4:tiny-h
ollama pull qwen2.5vl:3b

# 4. Flask uygulamasını başlat
python app_flask.py

# 5. Tarayıcıda aç
# http://localhost:5000
```

**İlk kullanım:** Video URL'si gir → Analiz et → **AI Video Q&A** butonuna tıkla → Keyfini çıkar! 🎉

## 🚀 Neler Yapabilirsiniz?

- 🎬 YouTube videolarını otomatik analiz edin
- 💬 **Video içeriği hakkında AI ile sohbet edin** (Türkçe!)
- 🔍 **Akıllı arama**: "Kod görseli göster", "Grafik var mı?" gibi sorgular
- 🖼️ **Her frame'i AI ile görsel analiz edin**
- ⏯️ **Arama sonuçlarına tıklayın, video o andan başlasın!**
- 📊 Detaylı istatistikler ve raporlar
- 🌐 Modern web arayüzü (Flask)

## 🛠️ Teknoloji Stack

- **Backend**: Python 3.x, Flask 3.0+
- **AI/LLM**: Ollama (Local)
  - `granite4:tiny-h` → Metin Q&A (2B parametre)
  - `qwen2.5vl:3b` → Görsel Analiz (Vision-Language)
- **Video İşleme**: OpenCV, PyTube
- **Transkript**: youtube-transcript-api
- **NLP**: NLTK
- **Frontend**: Bootstrap 5, jQuery
- **API**: YouTube IFrame API

## 🌟 Özellikler

### 🤖 **YENİ!** Yapay Zeka Özellikleri (Local Ollama)
- **🎯 Akıllı Video Soru-Cevap**: Video içeriği hakkında doğal dilde sorular sorun, AI size cevap versin!
  - � Gerçek zamanlı chat arayüzü
  - 🧠 Video transkriptinden otomatik context çıkarma
  - 🇹🇷 Türkçe dil desteği optimize edilmiş
  - ⚡ Hızlı yanıt süresi (granite4:tiny-h modeli)
  
- **🔍 Gelişmiş Akıllı Arama**: 3 farklı arama modu!
  - �📝 **Metin Arama**: Transkript içinde anahtar kelime araması
  - 🖼️ **Görsel Arama**: "Kod görseli göster", "Grafik var mı?" gibi sorgular
  - 🎨 **Hibrit Arama**: Metin + Görsel kombinasyonu (Optimize edilmiş!)
  - ⏱️ 60 saniyeden → 10-30 saniyeye düşürülmüş arama süresi
  - 🎯 Akıllı frame örnekleme stratejisi
  
- **👁️ Frame Görsel Analizi**: Her frame'i AI ile analiz edin
  - 🔬 Tek tıkla frame analizi
  - 🎨 Görsel içerik tespiti (qwen2.5vl:3b vision modeli)
  - 💡 Özel sorular sorabilme ("Bu görselde ne var?")
  
- **🎬 Akıllı Video Navigasyonu**:
  - 🖱️ Arama sonuçlarına tıklayın, video otomatik o anı açsın!
  - ⏯️ YouTube iframe entegrasyonu
  - ⚡ Otomatik oynatma ile zaman kaybı yok
  - 🎯 Frame'lerden direkt zaman damgası tespiti

### 📝 Metin Analizi
- **Tam Transkript**: Video'nun tüm konuşma metnini çıkarır
- **Zaman Damgalı Transkript**: Her cümlenin zamanını gösterir
- **Cümle Ayrıştırma**: Metni cümlelere ayırır
- **Çoklu Dil Desteği**: Türkçe ve İngilizce transkript desteği

### 🖼️ Görsel Analizi
- **Cümle Bitimi Frame'leri**: Her cümle veya soru bitiminde (. ? !) otomatik frame çıkarır
- **Düzenli Aralık Frame'leri**: Belirlediğiniz saniye aralıklarında frame çıkarır
- **Organize Klasör Yapısı**: Tüm görseller ayrı klasörde tutulur
- **🆕 AI Frame Analizi**: Her frame'i vision model ile analiz edebilme

### 📊 Video İstatistikleri
- Video başlığı, kanal adı
- Görüntülenme sayısı
- Video süresi
- Yayın tarihi
- Kelime ve karakter sayısı
- Frame istatistikleri

### 🌐 Web Arayüzü (Flask)
- **Responsive Tasarım**: Bootstrap 5 ile modern arayüz
- **Gerçek Zamanlı Chat**: AI ile interaktif sohbet
- **Frame Galerisi**: Tüm frame'leri görsel olarak gezinme
- **Akıllı Arama Paneli**: Metin, görsel ve hibrit arama seçenekleri
- **Gömülü Video Player**: YouTube videoları direkt arayüzde izleyin
- **Toast Bildirimleri**: Kullanıcı dostu geri bildirimler

## 📦 Kurulum

### 1️⃣ Python Gereksinimlerini Yükle

```bash
pip install -r requirements.txt
```

Veya manuel olarak:

```bash
pip install flask opencv-python youtube-transcript-api pytube nltk requests
```

### 2️⃣ Ollama'yı Kurun (AI Özellikleri İçin)

**Windows:**
1. [Ollama.com](https://ollama.com/download) adresinden indirin
2. Kurun ve otomatik başlatılacak

**Modelleri İndirin:**
```bash
ollama pull granite4:tiny-h
ollama pull qwen2.5vl:3b
```

### 3️⃣ Flask Web Arayüzünü Başlatın

```bash
python app_flask.py
```

Tarayıcınızda açın: `http://localhost:5000`

## 🚀 Kullanım

### 🌐 Web Arayüzü (Önerilen)

1. Flask uygulamasını başlatın: `python app_flask.py`
2. Tarayıcıda `http://localhost:5000` adresine gidin
3. YouTube URL'sini girin ve ayarları yapın
4. Video analiz edilsin!
5. **🆕 AI Video Q&A** butonuna tıklayarak:
   - Video hakkında sorular sorun 💬
   - Akıllı arama yapın 🔍
   - Frame'leri AI ile analiz edin 🖼️
   - Arama sonuçlarına tıklayarak videoyu o andan izleyin ▶️

### 💻 Komut Satırı Kullanımı

```bash
python youtube-app.py
```

Program sizden şunları soracak:
1. YouTube video URL'si
2. Düzenli aralıklarla frame çıkarılsın mı? (E/H)
3. Frame çıkarma aralığı (saniye)

### Örnek

```
YouTube video URL'sini girin: https://www.youtube.com/watch?v=dQw4w9WgXcQ

⚙️  Ayarlar:
Düzenli aralıklarla frame çıkarılsın mı? (E/H, varsayılan: E): E
Frame çıkarma aralığı (saniye, varsayılan: 30): 20
```

## 📁 Çıktı Yapısı

```
youtube_analysis/
└── {video_id}_{timestamp}/
    ├── images/
    │   ├── sentence_0000_time_00m15s.jpg
    │   ├── sentence_0001_time_00m32s.jpg
    │   ├── frame_0000_time_00m00s.jpg
    │   └── frame_0001_time_00m20s.jpg
    ├── text/
    │   ├── video_info.txt
    │   ├── full_transcript.txt
    │   ├── timed_transcript.txt
    │   ├── sentences.txt
    │   ├── sentence_frames_info.txt
    │   └── analysis_data.json
    ├── video.mp4
    └── SUMMARY_REPORT.txt
```

### Dosya Açıklamaları

#### 📂 images/ klasörü
- `sentence_XXXX_*.jpg`: Cümle bitimlerinde çıkarılan frame'ler
- `frame_XXXX_*.jpg`: Düzenli aralıklarla çıkarılan frame'ler

#### 📂 text/ klasörü
- `video_info.txt`: Video hakkında genel bilgiler
- `full_transcript.txt`: Tam transkript metni
- `timed_transcript.txt`: Zaman damgalı transkript
- `sentences.txt`: Cümlelere ayrılmış metin
- `sentence_frames_info.txt`: Her cümle frame'i hakkında bilgi
- `analysis_data.json`: Tüm verinin JSON formatı

#### 📄 Diğer dosyalar
- `video.mp4`: İndirilen video dosyası
- `SUMMARY_REPORT.txt`: Genel özet rapor

## 💡 Programatik Kullanım

```python
from youtube_app import YouTubeVideoAnalyzer

# Analyzer oluştur
analyzer = YouTubeVideoAnalyzer(
    url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    output_base_dir="my_analysis"
)

# Analizi çalıştır
analyzer.analyze(
    extract_interval_frames=True,  # Düzenli aralıklarla frame çıkar
    interval_seconds=30             # 30 saniyede bir frame
)
```

## 🔧 Özelleştirme

### Frame Çıkarma Aralığını Değiştir

```python
analyzer.analyze(interval_seconds=15)  # Her 15 saniyede bir
```

### Sadece Cümle Frame'leri

```python
analyzer.analyze(extract_interval_frames=False)  # Düzenli frame'leri devre dışı bırak
```

### Farklı Çıktı Klasörü

```python
analyzer = YouTubeVideoAnalyzer(
    url="your_url",
    output_base_dir="custom_output_folder"
)
```

## 📊 Örnek Çıktı

### Video Bilgileri
```
Başlık: Örnek Video Başlığı
Kanal: Kanal Adı
Görüntülenme: 1,234,567
Süre: 10:45
Yayın Tarihi: 2024-01-15
```

### Metin Analizi
```
Toplam Kelime Sayısı: 1,234
Toplam Karakter Sayısı: 7,890
Cümle Sayısı: 56
```

### Görsel Analizi
```
Cümle Bitimi Frame Sayısı: 56
Düzenli Aralık Frame Sayısı: 21
Toplam Frame Sayısı: 77
```

## ⚠️ Dikkat Edilmesi Gerekenler

1. **Transkript Durumu**: Bazı videolarda transkript olmayabilir
2. **Video İndirme**: Bazı videolar telif hakkı nedeniyle indirilemeyebilir
3. **İnternet Bağlantısı**: Video indirme için stabil internet gereklidir
4. **Disk Alanı**: Uzun videolar çok yer kaplayabilir
5. **🆕 Ollama Gereksinimleri**: AI özellikleri için Ollama sunucusu çalışıyor olmalı
   - Minimum 8GB RAM önerilir
   - GPU varsa çok daha hızlı çalışır
6. **🆕 İlk Kullanım**: Modeller ilk kullanımda indirilir (toplam ~4GB)

## 🐛 Hata Çözümleri

### "Transkript bulunamadı" hatası
- Video'da transkript olmayabilir
- Video gizli veya kısıtlı olabilir

### "Video indirilemedi" hatası
- İnternet bağlantınızı kontrol edin
- Video telif hakkı korumalı olabilir
- Farklı bir video URL'si deneyin

### 🆕 "Ollama'ya bağlanılamıyor" hatası
```bash
# Ollama servisini başlatın
ollama serve
```
Veya Windows'ta Ollama uygulamasının çalıştığından emin olun.

### 🆕 "Model bulunamadı" hatası
```bash
# Gerekli modelleri indirin
ollama pull granite4:tiny-h
ollama pull qwen2.5vl:3b
```

### 🆕 Yavaş Arama Sonuçları
- **Çözüm 1**: "Metin" veya "Hibrit" arama modunu kullanın (daha hızlı)
- **Çözüm 2**: Daha kısa videolarla test edin
- **Çözüm 3**: GPU'nuz varsa Ollama otomatik kullanacaktır

### NLTK veri hatası
İlk kullanımda otomatik olarak indirilir, manuel indirmek için:
```python
import nltk
nltk.download('punkt')
```

## 🎯 Performans İpuçları

### Arama Hızını Artırma
- **Hibrit Arama**: Önce metin ara, sonra sadece ilgili frame'leri analiz et (60-70% daha hızlı!)
- **Metin Arama**: Sadece transkript içinde arama yap (anında sonuç)
- **Görsel Arama**: 10 frame örnekleme ile optimize edildi (~20-30 saniye)

### AI Yanıt Kalitesini Artırma
- Spesifik sorular sorun: "Videoda neler anlatılıyor?" yerine "Video hangi teknolojiden bahsediyor?"
- Frame analizi kullanın: "Bu görselde ne var?" yerine "Bu görseldeki kod hangi dilde yazılmış?"

## 📝 Notlar

- Frame çıkarma işlemi video uzunluğuna göre zaman alabilir
- Yüksek çözünürlüklü videolar daha fazla disk alanı kaplar
- Cümle tespiti NLTK kütüphanesi ile yapılır (Türkçe ve İngilizce destekli)
- **🆕 AI özellikleri tamamen local çalışır** - verileriniz dışarı çıkmaz!
- **🆕 İlk model indirmesi** birkaç dakika sürebilir (sadece bir kez)

## 🎬 Özellik Gösterimleri

### 💬 AI Video Q&A
```
👤 "Bu videoda hangi konular anlatılıyor?"
🤖 "Videoda yapay zeka, makine öğrenmesi ve derin öğrenme konuları ele alınıyor..."

👤 "5. dakikada ne anlatılıyor?"
� "5. dakikada neural network mimarileri ve activation fonksiyonları detaylı olarak açıklanıyor..."
```

### 🔍 Akıllı Arama Örnekleri
- **Metin Arama**: "machine learning" → Tüm bahsedilen anlar listelensin
- **Görsel Arama**: "kod görseli" → Ekranda kod olan tüm anları bul
- **Hibrit Arama**: "python kod" → Hem "python" kelimesi geçen hem de kod görseli olan anlar

### 🖼️ Frame Analizi
```
🖼️ Frame'e tıkla → AI Analiz Et
🤖 "Bu görselde Python dilinde yazılmış bir for döngüsü ve liste comprehension örneği bulunuyor. 
    Kodda enumerate() fonksiyonu kullanılmış..."
```

### ⏯️ Akıllı Video Navigasyonu
```
🔍 Arama yap → Sonuç kartına tıkla → 🎬 Video otomatik o andan başlasın!
"Tam istediğim sahne!" ✅
```

## 🌟 Neden Bu Araç?

| Özellik | Geleneksel Yöntem | Bu Araç |
|---------|-------------------|---------|
| Video Arama | Manuel izleme, not alma | AI destekli, otomatik timestampler |
| Frame Çıkarma | Elle screenshot | Otomatik, zaman damgalı |
| İçerik Analizi | İzleyip not al | AI'ya sor, direkt cevap al |
| Görsel Arama | İmkansız | "Grafik göster" diye ara! |
| Gizlilik | Cloud servislere veri gönderme | %100 local, güvenli |
| Hız | Saatler | Dakikalar |

## 📊 Performans Karşılaştırması

**Önceki Versiyon vs Yeni Versiyon (AI Özellikli)**

- ❌ **Eski**: Frame analizi yok
- ✅ **Yeni**: AI ile görsel içerik analizi

- ❌ **Eski**: Manuel transkript okuma
- ✅ **Yeni**: "Ne anlatılıyor?" diye sor, AI cevaplasın

- ❌ **Eski**: Görsel arama yok
- ✅ **Yeni**: "Grafik göster" diye ara, AI bulsun

- ⏱️ **Arama Süresi**: 60-100 saniye → 10-30 saniye (**70% daha hızlı!**)

## �🤝 Katkıda Bulunma

Önerileriniz ve katkılarınız için GitHub üzerinden pull request gönderebilirsiniz.

### Geliştirme Fikirleri
- [ ] Batch video işleme
- [ ] Video karşılaştırma özelliği
- [ ] Bookmark/favori sistem
- [ ] Export Q&A history
- [ ] WebSocket ile real-time progress
- [ ] Daha fazla LLM model desteği
- [ ] Video özetleme özelliği

## 🎓 Kullanım Senaryoları

### 🎯 Eğitim & Öğrenme
- Ders videolarını analiz edin
- Spesifik konuları hızla bulun: "Bu videoda machine learning nerede anlatılıyor?"
- Kod örneklerini çıkarın: "Python kod örneklerini göster"

### 📊 İçerik Analizi
- Uzun röportajları analiz edin
- Ana konuları tespit edin
- Önemli anları bookmark'layın

### � Araştırma
- Teknik sunumları analiz edin
- Grafik ve diyagramları çıkarın: "Grafik göster"
- Belirli terimlerin geçtiği anları bulun

### 🎬 İçerik Üretimi
- Video scriptlerini çıkarın
- Önemli frame'leri thumbnail için kullanın
- Video içeriğini kategorize edin

## �📄 Lisans

Bu proje MIT lisansı altında lisanslanmıştır.

---

<div align="center">

### 🌟 Bu Projeyi Beğendiniz mi?

⭐ **Star verin** - Projenin gelişmesine destek olun!  
🐛 **Issue açın** - Hata bulduğunuzda bildirin  
🤝 **PR gönderin** - Katkıda bulunun  
💬 **Paylaşın** - Arkadaşlarınıza önerin  

---

**Yapımcı:** [Emre Developer](https://github.com/emredeveloper)  
**Teknoloji:** Python 🐍 | Flask 🌶️ | Ollama 🤖 | OpenCV 📹  
**Versiyon:** 2.0 (AI-Powered) 🚀  

---

*"AI ile videolarınızı daha akıllı analiz edin!"* ✨

</div>
