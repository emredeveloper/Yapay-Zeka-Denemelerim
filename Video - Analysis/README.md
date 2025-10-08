# 🎥 YouTube Video Analiz Aracı

YouTube videolarından transkript, görsel frame'ler ve istatistiksel bilgiler çıkaran kapsamlı bir Python aracı.

## 🌟 Özellikler

### 📝 Metin Analizi
- **Tam Transkript**: Video'nun tüm konuşma metnini çıkarır
- **Zaman Damgalı Transkript**: Her cümlenin zamanını gösterir
- **Cümle Ayrıştırma**: Metni cümlelere ayırır
- **Çoklu Dil Desteği**: Türkçe ve İngilizce transkript desteği

### 🖼️ Görsel Analizi
- **Cümle Bitimi Frame'leri**: Her cümle veya soru bitiminde (. ? !) otomatik frame çıkarır
- **Düzenli Aralık Frame'leri**: Belirlediğiniz saniye aralıklarında frame çıkarır
- **Organize Klasör Yapısı**: Tüm görseller ayrı klasörde tutulur

### 📊 Video İstatistikleri
- Video başlığı, kanal adı
- Görüntülenme sayısı
- Video süresi
- Yayın tarihi
- Kelime ve karakter sayısı
- Frame istatistikleri

## 📦 Kurulum

### Gereksinimleri Yükle

```bash
pip install -r requirements.txt
```

Veya manuel olarak:

```bash
pip install opencv-python youtube-transcript-api pytube nltk
```

## 🚀 Kullanım

### Temel Kullanım

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

## 🐛 Hata Çözümleri

### "Transkript bulunamadı" hatası
- Video'da transkript olmayabilir
- Video gizli veya kısıtlı olabilir

### "Video indirilemedi" hatası
- İnternet bağlantınızı kontrol edin
- Video telif hakkı korumalı olabilir
- Farklı bir video URL'si deneyin

### NLTK veri hatası
İlk kullanımda otomatik olarak indirilir, manuel indirmek için:
```python
import nltk
nltk.download('punkt')
```

## 📝 Notlar

- Frame çıkarma işlemi video uzunluğuna göre zaman alabilir
- Yüksek çözünürlüklü videolar daha fazla disk alanı kaplar
- Cümle tespiti NLTK kütüphanesi ile yapılır (Türkçe ve İngilizce destekli)

## 🤝 Katkıda Bulunma

Önerileriniz ve katkılarınız için GitHub üzerinden pull request gönderebilirsiniz.

## 📄 Lisans

Bu proje MIT lisansı altında lisanslanmıştır.

---

⭐ Beğendiyseniz yıldız vermeyi unutmayın!
