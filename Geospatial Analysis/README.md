# Türkiye Coğrafi Analiz ve Haritalama Uygulaması

Flask tabanlı web uygulaması ile Türkiye şehirlerini görselleştirin, mesafe hesaplayın ve analiz yapın.

## Özellikler

- 🗺️ **İnteraktif Harita**: Folium ile interaktif harita görselleştirme
- 📍 **20 Şehir Verisi**: Nüfus, alan, bölge, plaka kodu, yükseklik bilgileri
- 📏 **Mesafe Hesaplama**: Haversine formülü ile şehirler arası mesafe
- 📊 **İstatistikler**: Detaylı analiz ve istatistikler
- 📚 **Wikipedia Entegrasyonu**: Şehirler hakkında Wikipedia bilgileri
- 🎨 **Harita Stilleri**: OpenStreetMap, Satellite, Terrain, Dark
- 🔗 **Marker Clustering**: Çok sayıda şehir için otomatik gruplama
- 🤖 **AI Agent (LM Studio)**: Yerel LLM ile akıllı asistan
- 🛠️ **Tools & Agentic Flows**: Otomatik tool çağırma ile eylemler
- 🖼️ **Görsel Analiz**: Harita ve coğrafi görselleri analiz etme (VLM)

## Kurulum

1. Gerekli paketleri yükleyin:
```bash
pip install -r requirements.txt
```

2. LM Studio'yu kurun ve çalıştırın:
   - [LM Studio](https://lmstudio.ai/) indirip kurun
   - LM Studio'yu açın ve çalıştırın

3. VLM modelini indirin:
```bash
lms get qwen/qwen3-vl-4b
```

4. Uygulamayı çalıştırın:
```bash
python app_flask.py
```

5. Tarayıcınızda açın:
```
http://127.0.0.1:5000
```

**Not**: LM Studio'nun çalıştığından ve model'in yüklü olduğundan emin olun!

## Kullanım

1. **Şehir Seçimi**: Sol panelden şehirleri seçin
2. **Harita Oluşturma**: Ayarları yapıp "Harita Oluştur" butonuna tıklayın
3. **İstatistikler**: "İstatistikleri Hesapla" ile detaylı analiz görüntüleyin
4. **Mesafe Hesaplama**: "Mesafeleri Hesapla" ile şehirler arası mesafeleri görün
5. **Şehir Bilgileri**: Dropdown'dan şehir seçip Wikipedia bilgilerini görüntüleyin
6. **AI Agent Sohbet**: 
   - Doğal dil ile sorular sorun (örn: "Istanbul ve Ankara arasındaki mesafe nedir?")
   - Agent otomatik olarak uygun tools'ları kullanır
   - Örnek sorgular: Şehir bilgisi, mesafe, bölge arama, karşılaştırma
7. **Görsel Analiz**:
   - Harita veya coğrafi görsel yükleyin
   - Model görseli analiz edip şehirler, bölgeler ve coğrafi özellikler hakkında bilgi verir
   - Drag & drop ile görsel sürükleyip bırakabilirsiniz

## API Endpoints

- `GET /` - Ana sayfa
- `POST /api/create_map` - Harita oluştur
- `GET /api/city_info/<city_name>` - Şehir bilgisi
- `POST /api/statistics` - İstatistikler
- `POST /api/distances` - Mesafe hesaplama
- `POST /api/cities_table` - Şehir tablosu
- `POST /api/agent/chat` - AI Agent ile sohbet
- `GET /api/agent/health` - Agent durumu
- `POST /api/agent/analyze_image` - Görsel analiz (VLM)

## Dosya Yapısı

```
Geospatial Analysis/
├── app_flask.py          # Flask uygulaması
├── templates/
│   └── index.html        # Ana sayfa template
├── static/
│   ├── css/
│   │   └── style.css     # CSS stilleri
│   └── js/
│       └── main.js       # JavaScript kodları
└── requirements.txt      # Gerekli paketler
```

## AI Agent Tools

Agent şu tools'ları kullanabilir:
1. **get_city_info_tool**: Şehir detaylı bilgileri
2. **calculate_distance_tool**: İki şehir arası mesafe
3. **search_cities_tool**: Bölge/nüfus filtreleme ile şehir arama
4. **compare_cities_tool**: İki şehri karşılaştırma
5. **get_statistics_tool**: Çoklu şehir istatistikleri
6. **list_available_cities_tool**: Mevcut şehirler listesi
7. **analyze_map_image_tool**: Harita/coğrafi görsel analiz

## Notlar

- İlk çalıştırmada GeoJSON verileri yüklenir (biraz zaman alabilir)
- Wikipedia API'si kullanılıyor (internet bağlantısı gerekli)
- Harita interaktif olarak kullanılabilir (zoom, pan, marker tıklama)
- LM Studio çalışır durumda olmalı ve model yüklü olmalı
- Görsel analiz için VLM (Vision-Language Model) gerekli: `qwen/qwen3-vl-4b`
- Model indirme: `lms get qwen/qwen3-vl-4b`

