from flask import Flask, render_template, request, jsonify, send_file
import geopandas as gpd
from shapely.geometry import Point
import folium
from folium.plugins import MarkerCluster, MeasureControl, Draw
import requests
import pandas as pd
from math import radians, cos, sin, asin, sqrt
import json
import lmstudio as lms
from typing import List, Optional, Dict
import threading
from functools import wraps
import os
from werkzeug.utils import secure_filename
import base64
from datetime import datetime
from io import BytesIO
try:
    from PIL import Image, ImageDraw, ImageFont
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    print("[WARNING] PIL (Pillow) yüklü değil, harita görseli oluşturulamayacak")

try:
    import matplotlib
    matplotlib.use('Agg')  # GUI olmadan çalışması için
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("[WARNING] matplotlib yüklü değil, gelişmiş harita görseli oluşturulamayacak")

app = Flask(__name__)

# Dosya yükleme ayarları
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp', 'gif'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# LM Studio model instance (lazy loading)
_lm_model = None
_lm_lock = threading.Lock()

def get_lm_model():
    """LM Studio model instance'ını döndür (lazy loading) - VLM (Vision-Language Model)"""
    global _lm_model
    if _lm_model is None:
        with _lm_lock:
            if _lm_model is None:
                try:
                    # Vision-Language Model - görsel analiz desteği ile
                    model_name = "qwen/qwen3-vl-4b"
                    print(f"[LM Studio] Model yüklenmeye çalışılıyor: {model_name}")
                    print(f"[LM Studio] LM Studio bağlantısı kontrol ediliyor...")
                    
                    _lm_model = lms.llm(model_name)
                    
                    print(f"[LM Studio] ✓ Model başarıyla yüklendi: {model_name}")
                    print(f"[LM Studio] ✓ Görsel analiz özelliği aktif")
                    print(f"[LM Studio] ✓ Model hazır, istekler kabul ediliyor")
                    
                except ConnectionError as e:
                    print(f"[LM Studio] ✗ Bağlantı hatası: {e}")
                    print(f"[LM Studio] LM Studio'nun çalıştığından emin olun!")
                    print(f"[LM Studio] LM Studio'yu açın ve model'i yükleyin: {model_name}")
                    _lm_model = None
                except Exception as e:
                    import traceback
                    print(f"[LM Studio] ✗ Model yükleme hatası: {e}")
                    print(f"[LM Studio] Hata detayları:")
                    print(traceback.format_exc())
                    print(f"[LM Studio] Model'i indirmek için: lms get qwen/qwen3-vl-4b")
                    print(f"[LM Studio] LM Studio'nun çalıştığından ve model'in yüklü olduğundan emin olun.")
                    _lm_model = None
    else:
        print(f"[LM Studio] Model zaten yüklü, mevcut instance kullanılıyor")
    
    return _lm_model

# Türkiye şehir veritabanı (koordinatlar + sentetik veriler)
TURKIYE_SEHIRLER = {
    "Istanbul": {
        "koordinat": (41.0082, 28.9784),
        "nufus": 15519267,
        "alan_km2": 5461,
        "plaka": 34,
        "bolge": "Marmara",
        "yukseklik_m": 100,
        "wikipedia": "İstanbul"
    },
    "Ankara": {
        "koordinat": (39.9208, 32.8541),
        "nufus": 5663322,
        "alan_km2": 25632,
        "plaka": 6,
        "bolge": "İç Anadolu",
        "yukseklik_m": 938,
        "wikipedia": "Ankara"
    },
    "Izmir": {
        "koordinat": (38.4192, 27.1287),
        "nufus": 4425079,
        "alan_km2": 11973,
        "plaka": 35,
        "bolge": "Ege",
        "yukseklik_m": 2,
        "wikipedia": "İzmir"
    },
    "Antalya": {
        "koordinat": (36.8969, 30.7133),
        "nufus": 2606886,
        "alan_km2": 20791,
        "plaka": 7,
        "bolge": "Akdeniz",
        "yukseklik_m": 30,
        "wikipedia": "Antalya"
    },
    "Bursa": {
        "koordinat": (40.1826, 29.0665),
        "nufus": 3101836,
        "alan_km2": 10887,
        "plaka": 16,
        "bolge": "Marmara",
        "yukseklik_m": 100,
        "wikipedia": "Bursa"
    },
    "Adana": {
        "koordinat": (37.0000, 35.3213),
        "nufus": 2260323,
        "alan_km2": 14030,
        "plaka": 1,
        "bolge": "Akdeniz",
        "yukseklik_m": 23,
        "wikipedia": "Adana"
    },
    "Gaziantep": {
        "koordinat": (37.0662, 37.3833),
        "nufus": 2135258,
        "alan_km2": 6220,
        "plaka": 27,
        "bolge": "Güneydoğu Anadolu",
        "yukseklik_m": 850,
        "wikipedia": "Gaziantep"
    },
    "Konya": {
        "koordinat": (37.8713, 32.4846),
        "nufus": 2273168,
        "alan_km2": 38257,
        "plaka": 42,
        "bolge": "İç Anadolu",
        "yukseklik_m": 1016,
        "wikipedia": "Konya"
    },
    "Mersin": {
        "koordinat": (36.8000, 34.6333),
        "nufus": 1888581,
        "alan_km2": 15953,
        "plaka": 33,
        "bolge": "Akdeniz",
        "yukseklik_m": 10,
        "wikipedia": "Mersin"
    },
    "Kayseri": {
        "koordinat": (38.7312, 35.4787),
        "nufus": 1435475,
        "alan_km2": 17097,
        "plaka": 38,
        "bolge": "İç Anadolu",
        "yukseklik_m": 1054,
        "wikipedia": "Kayseri"
    },
    "Eskişehir": {
        "koordinat": (39.7767, 30.5206),
        "nufus": 898369,
        "alan_km2": 13925,
        "plaka": 26,
        "bolge": "İç Anadolu",
        "yukseklik_m": 792,
        "wikipedia": "Eskişehir"
    },
    "Diyarbakır": {
        "koordinat": (37.9144, 40.2306),
        "nufus": 1793241,
        "alan_km2": 15168,
        "plaka": 21,
        "bolge": "Güneydoğu Anadolu",
        "yukseklik_m": 674,
        "wikipedia": "Diyarbakır"
    },
    "Samsun": {
        "koordinat": (41.2867, 36.3300),
        "nufus": 1359680,
        "alan_km2": 9579,
        "plaka": 55,
        "bolge": "Karadeniz",
        "yukseklik_m": 4,
        "wikipedia": "Samsun"
    },
    "Denizli": {
        "koordinat": (37.7765, 29.0864),
        "nufus": 1053892,
        "alan_km2": 11868,
        "plaka": 20,
        "bolge": "Ege",
        "yukseklik_m": 354,
        "wikipedia": "Denizli"
    },
    "Şanlıurfa": {
        "koordinat": (37.1674, 38.7955),
        "nufus": 2143020,
        "alan_km2": 18584,
        "plaka": 63,
        "bolge": "Güneydoğu Anadolu",
        "yukseklik_m": 518,
        "wikipedia": "Şanlıurfa"
    },
    "Malatya": {
        "koordinat": (38.3552, 38.3095),
        "nufus": 808692,
        "alan_km2": 12235,
        "plaka": 44,
        "bolge": "Doğu Anadolu",
        "yukseklik_m": 954,
        "wikipedia": "Malatya"
    },
    "Kahramanmaraş": {
        "koordinat": (37.5858, 36.9371),
        "nufus": 1161984,
        "alan_km2": 14327,
        "plaka": 46,
        "bolge": "Akdeniz",
        "yukseklik_m": 568,
        "wikipedia": "Kahramanmaraş"
    },
    "Erzurum": {
        "koordinat": (39.9043, 41.2679),
        "nufus": 762062,
        "alan_km2": 25066,
        "plaka": 25,
        "bolge": "Doğu Anadolu",
        "yukseklik_m": 1890,
        "wikipedia": "Erzurum"
    },
    "Van": {
        "koordinat": (38.4891, 43.4089),
        "nufus": 1224689,
        "alan_km2": 19069,
        "plaka": 65,
        "bolge": "Doğu Anadolu",
        "yukseklik_m": 1726,
        "wikipedia": "Van"
    },
    "Batman": {
        "koordinat": (37.8812, 41.1351),
        "nufus": 634491,
        "alan_km2": 4637,
        "plaka": 72,
        "bolge": "Güneydoğu Anadolu",
        "yukseklik_m": 540,
        "wikipedia": "Batman"
    }
}

def get_wikipedia_info(city_name, lang="tr"):
    """Wikipedia'dan şehir bilgisi al"""
    try:
        wiki_name = TURKIYE_SEHIRLER.get(city_name, {}).get("wikipedia", city_name)
        url = f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/{wiki_name}"
        response = requests.get(url, timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            return {
                "ozet": data.get('extract', 'Bilgi bulunamadı.'),
                "baslik": data.get('title', city_name),
                "url": data.get('content_urls', {}).get('desktop', {}).get('page', ''),
                "resim": data.get('thumbnail', {}).get('source', '') if data.get('thumbnail') else ''
            }
        else:
            url_en = f"https://en.wikipedia.org/api/rest_v1/page/summary/{wiki_name}"
            response_en = requests.get(url_en, timeout=5)
            if response_en.status_code == 200:
                data = response_en.json()
                return {
                    "ozet": data.get('extract', 'Bilgi bulunamadı.'),
                    "baslik": data.get('title', city_name),
                    "url": data.get('content_urls', {}).get('desktop', {}).get('page', ''),
                    "resim": data.get('thumbnail', {}).get('source', '') if data.get('thumbnail') else ''
                }
    except:
        pass
    
    return {
        "ozet": f"{city_name} hakkında Wikipedia bilgisi alınamadı.",
        "baslik": city_name,
        "url": "",
        "resim": ""
    }

def load_turkey_geojson():
    """Türkiye il sınırlarını yükle"""
    urls = [
        "https://raw.githubusercontent.com/cihadturhan/tr-geojson/master/geo/tr-cities-utf8.json",
        "https://raw.githubusercontent.com/cihadturhan/tr-geojson/master/geo/tr.json",
        "https://raw.githubusercontent.com/cihadturhan/tr-geojson/master/geo/turkey.json"
    ]
    
    for url in urls:
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            geojson_data = response.json()
            turkey = gpd.GeoDataFrame.from_features(
                geojson_data.get('features', geojson_data if isinstance(geojson_data, list) else []), 
                crs="EPSG:4326"
            )
            if not turkey.empty:
                return turkey
        except:
            continue
    return None

def haversine(lon1, lat1, lon2, lat2):
    """İki nokta arasındaki mesafeyi km cinsinden hesapla"""
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371
    return c * r

def create_map_html(selected_cities, show_boundaries, show_clustering, map_style, show_lines):
    """Harita HTML oluştur"""
    if not selected_cities:
        return "<div class='alert alert-warning'>Lütfen en az bir şehir seçin!</div>"
    
    cities_to_show = {city: TURKIYE_SEHIRLER[city] for city in selected_cities if city in TURKIYE_SEHIRLER}
    
    if not cities_to_show:
        return "<div class='alert alert-warning'>Geçerli şehir seçilmedi!</div>"
    
    tile_options = {
        "OpenStreetMap": "OpenStreetMap",
        "Satellite": "Esri.WorldImagery",
        "Terrain": "OpenTopoMap",
        "Dark": "CartoDB.DarkMatter"
    }
    tiles = tile_options.get(map_style, "OpenStreetMap")
    
    center_lat = sum(data["koordinat"][0] for data in cities_to_show.values()) / len(cities_to_show)
    center_lon = sum(data["koordinat"][1] for data in cities_to_show.values()) / len(cities_to_show)
    
    m = folium.Map(location=[center_lat, center_lon], zoom_start=6, tiles=tiles)
    
    if show_clustering and len(cities_to_show) > 1:
        marker_cluster = MarkerCluster().add_to(m)
        marker_group = marker_cluster
    else:
        marker_group = m
    
    if show_boundaries:
        turkey = load_turkey_geojson()
        if turkey is not None and not turkey.empty:
            folium.GeoJson(
                turkey,
                name="Türkiye İl Sınırları",
                style_function=lambda feature: {
                    'fillColor': '#3388ff',
                    'color': 'black',
                    'weight': 2,
                    'fillOpacity': 0.1
                }
            ).add_to(m)
    
    for city, city_data in cities_to_show.items():
        lat, lon = city_data["koordinat"]
        
        popup_html = f"""
        <div style="min-width: 200px;">
            <h3 style="margin: 5px 0; color: #2c3e50;">{city}</h3>
            <hr style="margin: 5px 0;">
            <p style="margin: 3px 0;"><b>Bölge:</b> {city_data['bolge']}</p>
            <p style="margin: 3px 0;"><b>Nüfus:</b> {city_data['nufus']:,}</p>
            <p style="margin: 3px 0;"><b>Alan:</b> {city_data['alan_km2']:,} km²</p>
            <p style="margin: 3px 0;"><b>Yükseklik:</b> {city_data['yukseklik_m']} m</p>
            <p style="margin: 3px 0;"><b>Plaka:</b> {city_data['plaka']:02d}</p>
            <p style="margin: 3px 0; color: #7f8c8d; font-size: 11px;">Koordinat: ({lat:.4f}, {lon:.4f})</p>
        </div>
        """
        
        if city_data['nufus'] > 5000000:
            icon_color = 'darkred'
        elif city_data['nufus'] > 2000000:
            icon_color = 'red'
        elif city_data['nufus'] > 1000000:
            icon_color = 'orange'
        else:
            icon_color = 'blue'
        
        folium.Marker(
            location=[lat, lon],
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=f"{city} ({city_data['nufus']:,} nüfus)",
            icon=folium.Icon(color=icon_color, icon='info-sign')
        ).add_to(marker_group)
    
    if show_lines and len(cities_to_show) > 1:
        city_list = list(cities_to_show.items())
        for i in range(len(city_list)):
            for j in range(i + 1, len(city_list)):
                city1, data1 = city_list[i]
                city2, data2 = city_list[j]
                lat1, lon1 = data1["koordinat"]
                lat2, lon2 = data2["koordinat"]
                distance = haversine(lon1, lat1, lon2, lat2)
                
                folium.PolyLine(
                    locations=[[lat1, lon1], [lat2, lon2]],
                    color='blue',
                    weight=2,
                    opacity=0.5,
                    popup=f"<b>{city1} ↔ {city2}</b><br>Mesafe: {distance:.2f} km"
                ).add_to(m)
    
    folium.LayerControl().add_to(m)
    MeasureControl().add_to(m)
    Draw(export=True).add_to(m)
    
    return m._repr_html_()

def create_map_image(selected_cities, show_boundaries, show_lines, map_style='OpenStreetMap'):
    """Harita durumunu görsel olarak oluştur (PNG) - matplotlib ile"""
    try:
        if not HAS_MATPLOTLIB:
            # Fallback: PIL ile basit görsel
            return create_map_image_pil(selected_cities, show_boundaries, show_lines, map_style)
        
        cities_to_show = {city: TURKIYE_SEHIRLER[city] for city in selected_cities if city in TURKIYE_SEHIRLER}
        if not cities_to_show:
            return None
        
        # Matplotlib figure oluştur
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 10))
        fig.patch.set_facecolor('white')
        
        # Sol panel: Bilgiler
        ax1.axis('off')
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
        
        info_text = "Seçili Şehirler:\n\n"
        for i, (city, city_data) in enumerate(cities_to_show.items(), 1):
            lat, lon = city_data["koordinat"]
            info_text += f"{i}. {city} ({city_data['bolge']})\n"
            info_text += f"   Nüfus: {city_data['nufus']:,} | Alan: {city_data['alan_km2']:,} km²\n"
            info_text += f"   Koordinat: ({lat:.2f}°, {lon:.2f}°)\n\n"
        
        if len(cities_to_show) > 1:
            info_text += "\nMesafeler:\n\n"
            city_list = list(cities_to_show.items())
            for i in range(len(city_list)):
                for j in range(i + 1, len(city_list)):
                    city1, data1 = city_list[i]
                    city2, data2 = city_list[j]
                    lat1, lon1 = data1["koordinat"]
                    lat2, lon2 = data2["koordinat"]
                    distance = haversine(lon1, lat1, lon2, lat2)
                    info_text += f"   {city1} ↔ {city2}: {distance:.2f} km\n"
        
        ax1.text(0.05, 0.95, info_text, transform=ax1.transAxes, fontsize=11,
                verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Sağ panel: Harita
        ax2.set_title('Türkiye Harita Görünümü', fontsize=16, fontweight='bold')
        ax2.set_xlabel('Boylam (Longitude)', fontsize=12)
        ax2.set_ylabel('Enlem (Latitude)', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        # Türkiye koordinat aralığı
        lats = [data["koordinat"][0] for data in cities_to_show.values()]
        lons = [data["koordinat"][1] for data in cities_to_show.values()]
        
        min_lat, max_lat = min(lats) - 2, max(lats) + 2
        min_lon, max_lon = min(lons) - 2, max(lons) + 2
        
        # Türkiye sınırları (yaklaşık)
        turkey_lats = [36, 42, 42, 36, 36]
        turkey_lons = [26, 26, 45, 45, 26]
        ax2.plot(turkey_lons, turkey_lats, 'k-', linewidth=2, alpha=0.3, label='Türkiye Sınırları')
        
        # Şehirleri çiz
        colors = plt.cm.Set3(range(len(cities_to_show)))
        for i, (city, city_data) in enumerate(cities_to_show.items()):
            lat, lon = city_data["koordinat"]
            ax2.scatter(lon, lat, s=200, c=[colors[i]], marker='o', 
                       edgecolors='black', linewidths=2, label=city, zorder=5)
            ax2.annotate(city, (lon, lat), xytext=(5, 5), textcoords='offset points',
                        fontsize=10, fontweight='bold')
        
        # Mesafe çizgileri
        if show_lines and len(cities_to_show) > 1:
            city_list = list(cities_to_show.items())
            for i in range(len(city_list)):
                for j in range(i + 1, len(city_list)):
                    city1, data1 = city_list[i]
                    city2, data2 = city_list[j]
                    lat1, lon1 = data1["koordinat"]
                    lat2, lon2 = data2["koordinat"]
                    ax2.plot([lon1, lon2], [lat1, lat2], 'b--', alpha=0.5, linewidth=1.5)
        
        ax2.set_xlim(min_lon, max_lon)
        ax2.set_ylim(min_lat, max_lat)
        ax2.legend(loc='upper right', fontsize=9)
        ax2.set_aspect('equal', adjustable='box')
        
        # Byte stream'e kaydet
        img_bytes = BytesIO()
        plt.tight_layout()
        plt.savefig(img_bytes, format='png', dpi=100, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        img_bytes.seek(0)
        return img_bytes
        
    except Exception as e:
        print(f"[Map Image] matplotlib görsel oluşturma hatası: {e}")
        import traceback
        print(traceback.format_exc())
        # Fallback: PIL ile basit görsel
        return create_map_image_pil(selected_cities, show_boundaries, show_lines, map_style)

def create_map_image_pil(selected_cities, show_boundaries, show_lines, map_style='OpenStreetMap'):
    """Harita durumunu görsel olarak oluştur (PNG) - PIL ile fallback"""
    try:
        if not HAS_PIL:
            return None
        
        cities_to_show = {city: TURKIYE_SEHIRLER[city] for city in selected_cities if city in TURKIYE_SEHIRLER}
        if not cities_to_show:
            return None
        
        # Görsel boyutları
        width, height = 1200, 800
        img = Image.new('RGB', (width, height), color='white')
        draw = ImageDraw.Draw(img)
        
        # Basit font
        try:
            font_large = ImageFont.truetype("arial.ttf", 24)
            font_medium = ImageFont.truetype("arial.ttf", 18)
            font_small = ImageFont.truetype("arial.ttf", 14)
        except:
            font_large = ImageFont.load_default()
            font_medium = ImageFont.load_default()
            font_small = ImageFont.load_default()
        
        # Başlık
        title = "Türkiye Harita Görünümü"
        draw.text((width//2 - 150, 20), title, fill='black', font=font_large)
        
        # Şehir bilgileri
        y_offset = 80
        draw.text((50, y_offset), "📍 Seçili Şehirler:", fill='blue', font=font_medium)
        y_offset += 40
        
        for i, (city, city_data) in enumerate(cities_to_show.items(), 1):
            lat, lon = city_data["koordinat"]
            info_text = f"{i}. {city} ({city_data['bolge']})"
            draw.text((70, y_offset), info_text, fill='black', font=font_medium)
            y_offset += 30
            
            detail_text = f"   Nüfus: {city_data['nufus']:,} | Alan: {city_data['alan_km2']:,} km²"
            draw.text((70, y_offset), detail_text, fill='gray', font=font_small)
            y_offset += 35
        
        # Mesafeler
        if len(cities_to_show) > 1:
            y_offset += 20
            draw.text((50, y_offset), "📏 Mesafeler:", fill='blue', font=font_medium)
            y_offset += 40
            
            distances = []
            city_list = list(cities_to_show.items())
            for i in range(len(city_list)):
                for j in range(i + 1, len(city_list)):
                    city1, data1 = city_list[i]
                    city2, data2 = city_list[j]
                    lat1, lon1 = data1["koordinat"]
                    lat2, lon2 = data2["koordinat"]
                    distance = haversine(lon1, lat1, lon2, lat2)
                    distances.append((city1, city2, distance))
            
            for city1, city2, distance in distances[:5]:
                dist_text = f"   {city1} ↔ {city2}: {distance:.2f} km"
                draw.text((70, y_offset), dist_text, fill='black', font=font_small)
                y_offset += 25
        
        # Basit harita görseli (koordinat bazlı)
        map_area_x = width - 400
        map_area_y = 100
        map_area_w = 350
        map_area_h = 600
        
        draw.rectangle([map_area_x, map_area_y, map_area_x + map_area_w, map_area_y + map_area_h], 
                      outline='black', width=2)
        
        min_lat = min(data["koordinat"][0] for data in cities_to_show.values()) - 2
        max_lat = max(data["koordinat"][0] for data in cities_to_show.values()) + 2
        min_lon = min(data["koordinat"][1] for data in cities_to_show.values()) - 2
        max_lon = max(data["koordinat"][1] for data in cities_to_show.values()) + 2
        
        def lat_to_y(lat):
            return map_area_y + map_area_h - int((lat - min_lat) / (max_lat - min_lat) * map_area_h)
        
        def lon_to_x(lon):
            return map_area_x + int((lon - min_lon) / (max_lon - min_lon) * map_area_w)
        
        colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown', 'pink', 'cyan']
        for i, (city, city_data) in enumerate(cities_to_show.items()):
            lat, lon = city_data["koordinat"]
            x = lon_to_x(lon)
            y = lat_to_y(lat)
            color = colors[i % len(colors)]
            
            draw.ellipse([x-8, y-8, x+8, y+8], fill=color, outline='black', width=2)
            draw.text((x+12, y-10), city, fill='black', font=font_small)
        
        if show_lines and len(cities_to_show) > 1:
            city_list = list(cities_to_show.items())
            for i in range(len(city_list)):
                for j in range(i + 1, len(city_list)):
                    city1, data1 = city_list[i]
                    city2, data2 = city_list[j]
                    lat1, lon1 = data1["koordinat"]
                    lat2, lon2 = data2["koordinat"]
                    x1, y1 = lon_to_x(lon1), lat_to_y(lat1)
                    x2, y2 = lon_to_x(lon2), lat_to_y(lat2)
                    draw.line([x1, y1, x2, y2], fill='blue', width=2)
        
        img_bytes = BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        return img_bytes
        
    except Exception as e:
        print(f"[Map Image PIL] Görsel oluşturma hatası: {e}")
        import traceback
        print(traceback.format_exc())
        return None

@app.route('/api/map/image', methods=['POST'])
def api_map_image():
    """Harita durumunu görsel olarak oluştur ve döndür"""
    try:
        data = request.json
        selected_cities = data.get('cities', [])
        show_boundaries = data.get('show_boundaries', True)
        show_lines = data.get('show_lines', True)
        map_style = data.get('map_style', 'OpenStreetMap')
        
        if not selected_cities:
            return jsonify({'error': 'Şehir seçilmedi'}), 400
        
        img_bytes = create_map_image(selected_cities, show_boundaries, show_lines, map_style)
        
        if img_bytes is None:
            return jsonify({'error': 'Görsel oluşturulamadı'}), 500
        
        # Uploads klasörünün var olduğundan emin ol
        upload_folder = app.config['UPLOAD_FOLDER']
        os.makedirs(upload_folder, exist_ok=True)
        
        # Dosyaya kaydet
        filename = f"map_image_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        filepath = os.path.join(upload_folder, filename)
        
        with open(filepath, 'wb') as f:
            f.write(img_bytes.getvalue())
        
        # Base64 olarak döndür
        img_bytes.seek(0)
        img_base64 = base64.b64encode(img_bytes.read()).decode('utf-8')
        
        return jsonify({
            'success': True,
            'image_base64': f'data:image/png;base64,{img_base64}',
            'filename': filename
        })
        
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"[Map Image] HATA: {str(e)}")
        print(f"[Map Image] Traceback: {error_trace}")
        return jsonify({
            'error': f'Görsel oluşturma hatası: {str(e)}'
        }), 500

@app.route('/')
def index():
    """Ana sayfa"""
    return render_template('index.html', cities=list(TURKIYE_SEHIRLER.keys()))

@app.route('/api/create_map', methods=['POST'])
def api_create_map():
    """Harita oluşturma API"""
    data = request.json
    selected_cities = data.get('cities', [])
    show_boundaries = data.get('show_boundaries', True)
    show_clustering = data.get('show_clustering', False)
    map_style = data.get('map_style', 'OpenStreetMap')
    show_lines = data.get('show_lines', True)
    
    map_html = create_map_html(selected_cities, show_boundaries, show_clustering, map_style, show_lines)
    
    return jsonify({
        'success': True,
        'map_html': map_html,
        'message': f'Harita başarıyla oluşturuldu! {len(selected_cities)} şehir gösteriliyor.'
    })

@app.route('/api/city_info/<city_name>')
def api_city_info(city_name):
    """Şehir bilgisi API"""
    if city_name not in TURKIYE_SEHIRLER:
        return jsonify({'error': 'Şehir bulunamadı!'}), 404
    
    city_data = TURKIYE_SEHIRLER[city_name]
    lat, lon = city_data["koordinat"]
    wiki_info = get_wikipedia_info(city_name)
    
    return jsonify({
        'city': city_name,
        'data': city_data,
        'coordinates': {'lat': lat, 'lon': lon},
        'wikipedia': wiki_info
    })

@app.route('/api/statistics', methods=['POST'])
def api_statistics():
    """İstatistikler API"""
    data = request.json
    selected_cities = data.get('cities', [])
    
    if not selected_cities:
        return jsonify({'error': 'Lütfen şehir seçin!'}), 400
    
    cities_list = [city for city in selected_cities if city in TURKIYE_SEHIRLER]
    
    if len(cities_list) < 2:
        city_data = TURKIYE_SEHIRLER[cities_list[0]]
        lat, lon = city_data["koordinat"]
        return jsonify({
            'single_city': True,
            'city': cities_list[0],
            'data': city_data,
            'coordinates': {'lat': lat, 'lon': lon}
        })
    
    distances = []
    total_pop = 0
    total_area = 0
    
    for i in range(len(cities_list)):
        city_data = TURKIYE_SEHIRLER[cities_list[i]]
        total_pop += city_data['nufus']
        total_area += city_data['alan_km2']
        
        for j in range(i + 1, len(cities_list)):
            city1 = cities_list[i]
            city2 = cities_list[j]
            lat1, lon1 = TURKIYE_SEHIRLER[city1]["koordinat"]
            lat2, lon2 = TURKIYE_SEHIRLER[city2]["koordinat"]
            distance = haversine(lon1, lat1, lon2, lat2)
            distances.append({'city1': city1, 'city2': city2, 'distance': distance})
    
    if distances:
        max_dist = max(distances, key=lambda x: x['distance'])
        min_dist = min(distances, key=lambda x: x['distance'])
        avg_dist = sum(d['distance'] for d in distances) / len(distances)
        
        max_pop_city = max(cities_list, key=lambda c: TURKIYE_SEHIRLER[c]['nufus'])
        min_pop_city = min(cities_list, key=lambda c: TURKIYE_SEHIRLER[c]['nufus'])
        
        return jsonify({
            'single_city': False,
            'city_count': len(cities_list),
            'distance_pairs': len(distances),
            'max_distance': max_dist,
            'min_distance': min_dist,
            'avg_distance': avg_dist,
            'total_population': total_pop,
            'avg_population': total_pop / len(cities_list),
            'max_pop_city': {'name': max_pop_city, 'population': TURKIYE_SEHIRLER[max_pop_city]['nufus']},
            'min_pop_city': {'name': min_pop_city, 'population': TURKIYE_SEHIRLER[min_pop_city]['nufus']},
            'total_area': total_area,
            'avg_area': total_area / len(cities_list)
        })
    
    return jsonify({'error': 'İstatistik hesaplanamadı!'}), 400

@app.route('/api/distances', methods=['POST'])
def api_distances():
    """Mesafe hesaplama API"""
    data = request.json
    selected_cities = data.get('cities', [])
    
    if len(selected_cities) < 2:
        return jsonify({'error': 'En az 2 şehir seçmelisiniz!'}), 400
    
    distances = []
    cities_list = [city for city in selected_cities if city in TURKIYE_SEHIRLER]
    
    for i in range(len(cities_list)):
        for j in range(i + 1, len(cities_list)):
            city1 = cities_list[i]
            city2 = cities_list[j]
            lat1, lon1 = TURKIYE_SEHIRLER[city1]["koordinat"]
            lat2, lon2 = TURKIYE_SEHIRLER[city2]["koordinat"]
            distance = haversine(lon1, lat1, lon2, lat2)
            distances.append({
                'city1': city1,
                'city2': city2,
                'distance': round(distance, 2)
            })
    
    return jsonify({'distances': distances})

@app.route('/api/cities_table', methods=['POST'])
def api_cities_table():
    """Şehir tablosu API"""
    data = request.json
    selected_cities = data.get('cities', [])
    
    if not selected_cities:
        return jsonify({'cities': []})
    
    cities_list = [city for city in selected_cities if city in TURKIYE_SEHIRLER]
    
    table_data = []
    for city in cities_list:
        data_city = TURKIYE_SEHIRLER[city]
        table_data.append({
            "Şehir": city,
            "Bölge": data_city["bolge"],
            "Nüfus": f"{data_city['nufus']:,}",
            "Alan (km²)": f"{data_city['alan_km2']:,}",
            "Yükseklik (m)": data_city["yukseklik_m"],
            "Plaka": f"{data_city['plaka']:02d}"
        })
    
    return jsonify({'cities': table_data})

# ==================== LM Studio Agent Tools ====================

def get_city_info_tool(city_name: str) -> str:
    """Bir şehrin detaylı bilgilerini döndürür.
    
    Args:
        city_name: Şehir adı (örn: 'Istanbul', 'Ankara')
    
    Returns:
        Şehrin bilgilerini içeren JSON string
    """
    if city_name not in TURKIYE_SEHIRLER:
        available = ", ".join(list(TURKIYE_SEHIRLER.keys())[:5])
        return f"Hata: '{city_name}' şehri bulunamadı. Mevcut şehirler: {available}..."
    
    city_data = TURKIYE_SEHIRLER[city_name]
    lat, lon = city_data["koordinat"]
    wiki_info = get_wikipedia_info(city_name)
    
    result = {
        "şehir": city_name,
        "bölge": city_data['bolge'],
        "nüfus": city_data['nufus'],
        "alan_km2": city_data['alan_km2'],
        "yükseklik_m": city_data['yukseklik_m'],
        "plaka_kodu": f"{city_data['plaka']:02d}",
        "koordinat": {"lat": lat, "lon": lon},
        "wikipedia_özet": wiki_info['ozet'][:200] + "..." if len(wiki_info['ozet']) > 200 else wiki_info['ozet']
    }
    
    return json.dumps(result, ensure_ascii=False, indent=2)

def calculate_distance_tool(city1: str, city2: str, calculate_time: bool = True) -> str:
    """İki şehir arasındaki mesafeyi km cinsinden hesaplar ve araba ile süreyi de hesaplar.
    
    Args:
        city1: İlk şehir adı
        city2: İkinci şehir adı
        calculate_time: Araba ile süre hesaplansın mı (varsayılan: True)
    
    Returns:
        Mesafe ve süre bilgisini içeren JSON string
    """
    if city1 not in TURKIYE_SEHIRLER:
        return json.dumps({"hata": f"'{city1}' şehri bulunamadı."}, ensure_ascii=False)
    if city2 not in TURKIYE_SEHIRLER:
        return json.dumps({"hata": f"'{city2}' şehri bulunamadı."}, ensure_ascii=False)
    
    lat1, lon1 = TURKIYE_SEHIRLER[city1]["koordinat"]
    lat2, lon2 = TURKIYE_SEHIRLER[city2]["koordinat"]
    distance_km = haversine(lon1, lat1, lon2, lat2)
    
    # Araba ile süre hesapla (ortalama 100 km/saat)
    avg_speed_kmh = 100
    time_hours = distance_km / avg_speed_kmh
    time_hours_int = int(time_hours)
    time_minutes = int((time_hours - time_hours_int) * 60)
    
    result = {
        "şehir1": city1,
        "şehir2": city2,
        "mesafe_km": round(distance_km, 2),
        "araba_ile_süre": {
            "saat": time_hours_int,
            "dakika": time_minutes,
            "toplam_saat": round(time_hours, 1)
        }
    }
    
    return json.dumps(result, ensure_ascii=False, indent=2)

def search_cities_tool(bolge: Optional[str] = None, min_nufus: Optional[int] = None, 
                       max_nufus: Optional[int] = None) -> str:
    """Şehirleri filtreleyerek arama yapar.
    
    Args:
        bolge: Bölge adı (örn: 'Marmara', 'İç Anadolu')
        min_nufus: Minimum nüfus
        max_nufus: Maximum nüfus
    
    Returns:
        Bulunan şehirlerin listesi
    """
    results = []
    
    for city, data in TURKIYE_SEHIRLER.items():
        if bolge and data['bolge'] != bolge:
            continue
        if min_nufus and data['nufus'] < min_nufus:
            continue
        if max_nufus and data['nufus'] > max_nufus:
            continue
        
        results.append({
            "şehir": city,
            "bölge": data['bolge'],
            "nüfus": data['nufus'],
            "alan_km2": data['alan_km2']
        })
    
    if not results:
        return "Hiç şehir bulunamadı."
    
    return json.dumps(results, ensure_ascii=False, indent=2)

def compare_cities_tool(city1: str, city2: str) -> str:
    """İki şehri karşılaştırır.
    
    Args:
        city1: İlk şehir adı
        city2: İkinci şehir adı
    
    Returns:
        Karşılaştırma sonuçlarını içeren string
    """
    if city1 not in TURKIYE_SEHIRLER or city2 not in TURKIYE_SEHIRLER:
        return "Hata: Şehirlerden biri bulunamadı."
    
    data1 = TURKIYE_SEHIRLER[city1]
    data2 = TURKIYE_SEHIRLER[city2]
    
    comparison = {
        "şehirler": [city1, city2],
        "nüfus_karşılaştırma": {
            city1: data1['nufus'],
            city2: data2['nufus'],
            "fark": abs(data1['nufus'] - data2['nufus']),
            "daha_kalabalık": city1 if data1['nufus'] > data2['nufus'] else city2
        },
        "alan_karşılaştırma": {
            city1: data1['alan_km2'],
            city2: data2['alan_km2'],
            "fark_km2": abs(data1['alan_km2'] - data2['alan_km2']),
            "daha_geniş": city1 if data1['alan_km2'] > data2['alan_km2'] else city2
        },
        "mesafe_km": round(haversine(
            data1["koordinat"][1], data1["koordinat"][0],
            data2["koordinat"][1], data2["koordinat"][0]
        ), 2)
    }
    
    return json.dumps(comparison, ensure_ascii=False, indent=2)

def get_statistics_tool(cities: List[str]) -> str:
    """Seçilen şehirler için istatistikler hesaplar.
    
    Args:
        cities: Şehir adları listesi
    
    Returns:
        İstatistikleri içeren string
    """
    if not cities:
        return "Hata: En az bir şehir seçmelisiniz."
    
    valid_cities = [c for c in cities if c in TURKIYE_SEHIRLER]
    if not valid_cities:
        return "Hata: Geçerli şehir bulunamadı."
    
    if len(valid_cities) == 1:
        city_data = TURKIYE_SEHIRLER[valid_cities[0]]
        return json.dumps({
            "şehir": valid_cities[0],
            "nüfus": city_data['nufus'],
            "alan_km2": city_data['alan_km2'],
            "bölge": city_data['bolge']
        }, ensure_ascii=False, indent=2)
    
    # Çoklu şehir istatistikleri
    distances = []
    total_pop = 0
    total_area = 0
    
    for i in range(len(valid_cities)):
        city_data = TURKIYE_SEHIRLER[valid_cities[i]]
        total_pop += city_data['nufus']
        total_area += city_data['alan_km2']
        
        for j in range(i + 1, len(valid_cities)):
            city1 = valid_cities[i]
            city2 = valid_cities[j]
            lat1, lon1 = TURKIYE_SEHIRLER[city1]["koordinat"]
            lat2, lon2 = TURKIYE_SEHIRLER[city2]["koordinat"]
            distance = haversine(lon1, lat1, lon2, lat2)
            distances.append({
                "şehir1": city1,
                "şehir2": city2,
                "mesafe_km": round(distance, 2)
            })
    
    stats = {
        "şehir_sayısı": len(valid_cities),
        "toplam_nüfus": total_pop,
        "ortalama_nüfus": round(total_pop / len(valid_cities)),
        "toplam_alan_km2": total_area,
        "ortalama_alan_km2": round(total_area / len(valid_cities)),
        "mesafeler": distances
    }
    
    if distances:
        max_dist = max(distances, key=lambda x: x['mesafe_km'])
        min_dist = min(distances, key=lambda x: x['mesafe_km'])
        stats["en_uzak_şehirler"] = max_dist
        stats["en_yakın_şehirler"] = min_dist
        stats["ortalama_mesafe_km"] = round(sum(d['mesafe_km'] for d in distances) / len(distances), 2)
    
    return json.dumps(stats, ensure_ascii=False, indent=2)

def list_available_cities_tool() -> str:
    """Mevcut tüm şehirlerin listesini döndürür.
    
    Returns:
        Şehir listesi
    """
    cities_list = [{"şehir": city, "bölge": data["bolge"]} 
                   for city, data in TURKIYE_SEHIRLER.items()]
    return json.dumps(cities_list, ensure_ascii=False, indent=2)

def analyze_map_image_tool(image_path: str, question: Optional[str] = None) -> str:
    """Harita veya coğrafi görsel analiz eder ve sorularınızı yanıtlar.
    
    Args:
        image_path: Analiz edilecek görsel dosyasının yolu
        question: Görsel hakkında sorulacak soru (opsiyonel)
    
    Returns:
        Görsel analiz sonuçlarını içeren string
    """
    try:
        model = get_lm_model()
        if model is None:
            return "Hata: LM Studio model yüklenemedi."
        
        # Görseli hazırla
        image_handle = lms.prepare_image(image_path)
        
        # Soru belirtilmemişse varsayılan soru kullan
        if not question:
            question = "Bu harita veya coğrafi görseli analiz et. Hangi şehirler, bölgeler veya coğrafi özellikler görünüyor? Detaylı bir açıklama yap."
        
        # Chat oluştur ve görseli ekle
        chat = lms.Chat()
        chat.add_user_message(question, images=[image_handle])
        
        # Model yanıtı
        prediction = model.respond(chat)
        
        # Yanıtı extract et
        if hasattr(prediction, 'content'):
            content = prediction.content
            if isinstance(content, list):
                # TextData listesi
                text_parts = []
                for item in content:
                    if hasattr(item, 'text'):
                        text_parts.append(item.text)
                    elif isinstance(item, str):
                        text_parts.append(item)
                return '\n'.join(text_parts) if text_parts else str(prediction)
            elif isinstance(content, str):
                return content
            else:
                return str(content)
        else:
            return str(prediction)
    
    except Exception as e:
        return f"Hata: Görsel analiz edilemedi. {str(e)}"

def get_current_map_info_tool(selected_cities: Optional[List[str]] = None, 
                                map_style: Optional[str] = None,
                                show_boundaries: bool = True,
                                show_clustering: bool = False,
                                show_lines: bool = True) -> str:
    """Mevcut harita durumunu ve seçili şehirleri analiz eder.
    Bu tool, kullanıcının haritada ne gösterdiğini anlamak için kullanılır.
    
    Args:
        selected_cities: Seçili şehirlerin listesi
        map_style: Harita stili (OpenStreetMap, Satellite, Terrain, Dark)
        show_boundaries: İl sınırlarını göster
        show_clustering: Marker clustering aktif
        show_lines: Mesafe çizgilerini göster
    
    Returns:
        Mevcut harita durumunu içeren detaylı string
    """
    try:
        if not selected_cities or len(selected_cities) == 0:
            return json.dumps({
                "durum": "Harita boş",
                "mesaj": "Haritada henüz şehir seçilmemiş. Kullanıcı şehir seçip harita oluşturmalı."
            }, ensure_ascii=False, indent=2)
        
        # Seçili şehirlerin detaylı bilgilerini topla
        cities_info = []
        total_pop = 0
        total_area = 0
        regions = set()
        
        for city in selected_cities:
            if city in TURKIYE_SEHIRLER:
                city_data = TURKIYE_SEHIRLER[city]
                lat, lon = city_data["koordinat"]
                cities_info.append({
                    "şehir": city,
                    "bölge": city_data["bolge"],
                    "nüfus": city_data["nufus"],
                    "alan_km2": city_data["alan_km2"],
                    "yükseklik_m": city_data["yukseklik_m"],
                    "koordinat": {"lat": lat, "lon": lon}
                })
                total_pop += city_data["nufus"]
                total_area += city_data["alan_km2"]
                regions.add(city_data["bolge"])
        
        # Mesafeleri hesapla
        distances_info = []
        if len(cities_info) > 1:
            for i in range(len(cities_info)):
                for j in range(i + 1, len(cities_info)):
                    city1 = cities_info[i]
                    city2 = cities_info[j]
                    lat1, lon1 = city1["koordinat"]["lat"], city1["koordinat"]["lon"]
                    lat2, lon2 = city2["koordinat"]["lat"], city2["koordinat"]["lon"]
                    distance = haversine(lon1, lat1, lon2, lat2)
                    distances_info.append({
                        "şehir1": city1["şehir"],
                        "şehir2": city2["şehir"],
                        "mesafe_km": round(distance, 2)
                    })
        
        # Harita durumu özeti
        map_info = {
            "harita_durumu": "Aktif",
            "seçili_şehir_sayısı": len(cities_info),
            "şehirler": cities_info,
            "toplam_nüfus": total_pop,
            "toplam_alan_km2": total_area,
            "ortalama_nüfus": round(total_pop / len(cities_info)) if cities_info else 0,
            "ortalama_alan_km2": round(total_area / len(cities_info)) if cities_info else 0,
            "bölgeler": list(regions),
            "harita_stili": map_style or "OpenStreetMap",
            "ayarlar": {
                "il_sınırları": "Gösteriliyor" if show_boundaries else "Gizli",
                "marker_clustering": "Aktif" if show_clustering else "Pasif",
                "mesafe_çizgileri": "Gösteriliyor" if show_lines else "Gizli"
            }
        }
        
        if distances_info:
            max_dist = max(distances_info, key=lambda x: x["mesafe_km"])
            min_dist = min(distances_info, key=lambda x: x["mesafe_km"])
            map_info["mesafeler"] = distances_info
            map_info["en_uzak_şehirler"] = max_dist
            map_info["en_yakın_şehirler"] = min_dist
            map_info["ortalama_mesafe_km"] = round(
                sum(d["mesafe_km"] for d in distances_info) / len(distances_info), 2
            )
        
        # JSON formatında döndür - tool olarak kullanılacak
        return json.dumps(map_info, ensure_ascii=False, indent=2)
        
    except Exception as e:
        return f"Hata: Harita bilgisi alınamadı. {str(e)}"

# Tools listesi
AGENT_TOOLS = [
    get_city_info_tool,
    calculate_distance_tool,
    search_cities_tool,
    compare_cities_tool,
    get_statistics_tool,
    list_available_cities_tool,
    analyze_map_image_tool,
    get_current_map_info_tool
]

@app.route('/api/agent/chat', methods=['POST'])
def api_agent_chat():
    """LM Studio Agent ile sohbet endpoint'i"""
    try:
        data = request.json
        if not data:
            return jsonify({'error': 'Geçersiz istek formatı!'}), 400
        
        user_message = data.get('message', '')
        conversation_history = data.get('history', [])
        
        # Harita durumu bilgilerini al (eğer gönderilmişse)
        map_state = data.get('map_state', {})
        selected_cities = map_state.get('cities', [])
        map_style = map_state.get('map_style', 'OpenStreetMap')
        show_boundaries = map_state.get('show_boundaries', True)
        show_clustering = map_state.get('show_clustering', False)
        show_lines = map_state.get('show_lines', True)
        
        if not user_message:
            return jsonify({'error': 'Mesaj boş olamaz!'}), 400
        
        print(f"[Agent Chat] Mesaj alındı: {user_message[:50]}...")
        if selected_cities:
            print(f"[Agent Chat] Harita durumu: {len(selected_cities)} şehir seçili")
        
        model = get_lm_model()
        if model is None:
            error_msg = 'LM Studio model yüklenemedi. LM Studio\'nun çalıştığından ve model\'in yüklü olduğundan emin olun. Model: qwen/qwen3-vl-4b'
            print(f"[Agent Chat] HATA: {error_msg}")
            return jsonify({
                'error': error_msg,
                'suggestion': 'LM Studio\'yu açın ve "qwen/qwen3-vl-4b" modelini yükleyin. Model yoksa: lms get qwen/qwen3-vl-4b'
            }), 503
        
        print(f"[Agent Chat] Model hazır, sohbet başlatılıyor...")
        
        # Chat instance oluştur
        system_prompt = """Sen Türkiye şehirleri hakkında yardımcı bir coğrafi analiz asistanısın. 
Kullanıcılara şehir bilgileri, mesafe hesaplama, istatistikler ve karşılaştırmalar konusunda yardımcı ol.

ÖNEMLİ KURALLAR:
1. Yanıtlarını KISA ve ÖZ tut - uzun listeler yapma
2. Kullanıcının sorusunu direkt yanıtla
3. Mesafe sorusu sorulursa, araba ile süre hesapla (ortalama 100 km/saat hız varsay)
4. Harita sorusu sorulursa, get_current_map_info_tool fonksiyonunu kullan
5. Türkçe yanıt ver, samimi ve anlaşılır dil kullan
6. Formatlanmış metin kullan (\n yerine düzgün satırlar)
7. Kullanıcıyı tatmin et, sadece bilgi yığını değil"""
        
        chat = lms.Chat(system_prompt)
        
        # Konuşma geçmişini ekle (önce)
        # NOT: LM Studio Chat objesi sadece user mesajlarını ekler, assistant mesajları agent tarafından üretilir
        for msg in conversation_history:
            if msg.get('role') == 'user':
                chat.add_user_message(msg.get('content', ''))
            # Assistant mesajlarını eklemiyoruz - agent kendi yanıtlarını üretecek
            # LM Studio Chat objesinde add_assistant_message metodu yok
        
        # Chat'e mesaj ekle - harita durumu context'i tool'lar üzerinden sağlanacak
        chat.add_user_message(user_message)
        
        print(f"[Agent Chat] Chat hazır, {len(AGENT_TOOLS)} tool mevcut")
        
        # Agent yanıtını topla
        assistant_messages = []
        tool_call_count = 0
        final_response = None
        
        def extract_content(obj):
            """TextData veya diğer objelerden string içerik çıkar"""
            if obj is None:
                return None
            
            # String ise direkt döndür
            if isinstance(obj, str):
                return obj
            
            # Liste ise birleştir
            if isinstance(obj, (list, tuple)):
                parts = []
                for item in obj:
                    extracted = extract_content(item)
                    if extracted:
                        parts.append(extracted)
                if parts:
                    return '\n'.join(parts)
                return None
            
            # TextData objesi kontrolü - öncelikli
            obj_type = type(obj).__name__
            if 'TextData' in obj_type:
                # TextData için 'text' attribute'unu kontrol et (en yaygın)
                if hasattr(obj, 'text'):
                    text_val = obj.text
                    if isinstance(text_val, str) and text_val:
                        return text_val
                    elif text_val:
                        return str(text_val)
                # Diğer olası attribute'lar
                for attr in ['data', 'content', 'value', 'message']:
                    if hasattr(obj, attr):
                        val = getattr(obj, attr)
                        if isinstance(val, str) and val:
                            return val
                        elif val:
                            str_val = str(val)
                            if str_val and not str_val.startswith('<'):
                                return str_val
            
            # Text veya Message gibi objeler için
            if 'Text' in obj_type or 'Message' in obj_type:
                for attr in ['text', 'data', 'content', 'value', 'message']:
                    if hasattr(obj, attr):
                        val = getattr(obj, attr)
                        if isinstance(val, str) and val:
                            return val
                        elif val:
                            str_val = str(val)
                            if str_val and not str_val.startswith('<'):
                                return str_val
            
            # Content attribute kontrolü
            if hasattr(obj, 'content'):
                content = obj.content
                if isinstance(content, str):
                    return content
                # TextData içinde nested olabilir
                elif content is not None:
                    nested_type = type(content).__name__
                    if 'TextData' in nested_type:
                        # Nested TextData için text attribute'unu kontrol et
                        if hasattr(content, 'text'):
                            text_val = content.text
                            if isinstance(text_val, str) and text_val:
                                return text_val
                        # Diğer attribute'ları dene
                        for attr in ['data', 'value', 'message']:
                            if hasattr(content, attr):
                                val = getattr(content, attr)
                                if isinstance(val, str) and val:
                                    return val
                                elif val:
                                    str_val = str(val)
                                    if str_val and not str_val.startswith('<'):
                                        return str_val
                    # String'e çevirmeyi dene
                    try:
                        str_content = str(content)
                        if str_content and not str_content.startswith('<') and len(str_content) > 5:
                            return str_content
                    except:
                        pass
            
            # Son çare: objeyi string'e çevir
            try:
                result = str(obj)
                # Eğer object representation ise None döndür
                if result.startswith('<') and result.endswith('>'):
                    # Ancak içinde text= varsa parse etmeyi dene
                    if 'text=' in result:
                        import re
                        match = re.search(r'text=["\']([^"\']+)["\']', result)
                        if match:
                            return match.group(1)
                    return None
                # Boş değilse döndür
                return result if result and len(result) > 0 else None
            except:
                return None
        
        def on_message(message):
            """Her mesaj için callback - chat'e ekle ve topla"""
            nonlocal final_response
            # Chat'e ekle (LM Studio API gereksinimi)
            chat.append(message)
            
            # Assistant mesajlarını topla
            try:
                content = None
                
                # Önce message.content'i kontrol et (liste olabilir)
                if hasattr(message, 'content'):
                    message_content = message.content
                    
                    # Eğer content bir liste ise (TextData listesi olabilir)
                    if isinstance(message_content, list):
                        print(f"[Agent Chat] Content bir liste, uzunluk: {len(message_content)}")
                        # Her bir item'ı extract_content ile işle
                        text_parts = []
                        for item in message_content:
                            extracted = extract_content(item)
                            if extracted:
                                text_parts.append(extracted)
                        
                        if text_parts:
                            content = '\n'.join(text_parts)
                            print(f"[Agent Chat] Liste içeriği birleştirildi: {len(content)} karakter")
                    else:
                        # Tek bir obje ise direkt extract_content kullan
                        content = extract_content(message_content)
                
                # Eğer content hala None ise, message objesinin kendisini dene
                if not content or len(content.strip()) == 0:
                    if hasattr(message, 'role') and message.role == 'assistant':
                        # Assistant mesajı için message objesini parse et
                        content = extract_content(message)
                        if not content:
                            # Message'ın tüm attribute'larını kontrol et
                            for attr in ['text', 'data', 'value', 'message']:
                                if hasattr(message, attr):
                                    val = getattr(message, attr)
                                    content = extract_content(val)
                                    if content:
                                        break
                
                # Eğer hala content yoksa, message objesinin kendisini string'e çevir
                if not content or len(content.strip()) == 0:
                    # Son çare: message'ı direkt string'e çevir ve parse et
                    msg_str = str(message)
                    if 'TextData' in msg_str and 'text=' in msg_str:
                        import re
                        # TextData(text="...") formatından text'i çıkar
                        matches = re.findall(r'text=["\']([^"\']+)["\']', msg_str)
                        if matches:
                            content = '\n'.join(matches)
                            print(f"[Agent Chat] TextData string'den parse edildi: {len(content)} karakter")
                
                # Content bulunduysa ekle
                if content and len(content.strip()) > 0:
                    assistant_messages.append(content)
                    # En uzun yanıtı final_response olarak sakla
                    if not final_response or len(content) > len(final_response):
                        final_response = content
                    print(f"[Agent Chat] ✓ Mesaj içeriği alındı: {len(content)} karakter")
                else:
                    # Debug bilgisi
                    msg_type = type(message).__name__
                    print(f"[Agent Chat] ⚠ Mesaj parse edilemedi, type: {msg_type}")
                    if hasattr(message, 'content'):
                        content_type = type(message.content).__name__
                        print(f"[Agent Chat] ⚠ Content type: {content_type}")
                        if isinstance(message.content, list) and message.content:
                            first_item_type = type(message.content[0]).__name__
                            print(f"[Agent Chat] ⚠ First item type: {first_item_type}")
                            if hasattr(message.content[0], 'text'):
                                print(f"[Agent Chat] ⚠ First item.text value: {getattr(message.content[0], 'text', 'N/A')[:100]}")
                    
            except Exception as e:
                print(f"[Agent Chat] Mesaj parse hatası: {e}")
                import traceback
                print(traceback.format_exc())
        
        def on_prediction_fragment(fragment, round_index=0):
            """Streaming için callback"""
            pass
        
        def on_prediction_completed(result, round_index=0):
            """Prediction tamamlandığında"""
            nonlocal final_response
            print(f"[Agent Chat] Round {round_index} prediction tamamlandı")
            try:
                content = extract_content(result)
                if content:
                    assistant_messages.append(content)
                    if not final_response or len(content) > len(final_response):
                        final_response = content
                    print(f"[Agent Chat] Prediction içeriği: {len(content)} karakter")
            except Exception as e:
                print(f"[Agent Chat] Prediction parse hatası: {e}")
                import traceback
                print(traceback.format_exc())
        
        def on_round_end(round_index):
            """Round sonu callback"""
            nonlocal tool_call_count
            tool_call_count += 1
            print(f"[Agent Chat] Round {round_index} tamamlandı")
        
        print(f"[Agent Chat] Agent.act() çağrılıyor...")
        
        # Agent'i çalıştır
        model.act(
            chat,
            AGENT_TOOLS,
            on_message=on_message,
            on_prediction_fragment=on_prediction_fragment,
            on_prediction_completed=on_prediction_completed,
            on_round_end=on_round_end
        )
        
        print(f"[Agent Chat] Agent.act() tamamlandı, {len(assistant_messages)} assistant mesajı toplandı")
        
        # Son yanıtı al - en uzun yanıtı tercih et
        assistant_message = None
        
        if assistant_messages:
            # Sadece string olanları filtrele ve en uzun yanıtı al
            string_messages = [msg for msg in assistant_messages if isinstance(msg, str) and len(msg) > 0]
            if string_messages:
                assistant_message = max(string_messages, key=len)
                print(f"[Agent Chat] En uzun yanıt listeden alındı: {len(assistant_message)} karakter")
            else:
                # String olmayan mesajları string'e çevir
                assistant_message = str(assistant_messages[-1]) if assistant_messages else None
                print(f"[Agent Chat] Yanıt string'e çevrildi: {len(assistant_message) if assistant_message else 0} karakter")
        elif final_response:
            # final_response'u string'e çevir
            if isinstance(final_response, str):
                assistant_message = final_response
            else:
                # TextData veya diğer objeler için extract_content kullan
                assistant_message = extract_content(final_response) or str(final_response)
            print(f"[Agent Chat] Final yanıt callback'den alındı, uzunluk: {len(assistant_message) if assistant_message else 0} karakter")
        else:
            # Eğer mesaj toplanamadıysa, chat'ten almaya çalış
            try:
                # Chat objesinin iç yapısını kontrol et
                if hasattr(chat, '_messages'):
                    messages = chat._messages
                    print(f"[Agent Chat] Chat._messages mevcut, {len(messages)} mesaj")
                    for msg in reversed(messages):
                        content = extract_content(msg)
                        if content and len(content) > 10:  # En az 10 karakterlik yanıt
                            assistant_message = content
                            print(f"[Agent Chat] Yanıt _messages'dan alındı: {len(assistant_message)} karakter")
                            break
                    
                    if not assistant_message:
                        assistant_message = "Yanıt oluşturulamadı - mesaj bulunamadı."
                elif hasattr(chat, 'messages'):
                    # Bazı versiyonlarda messages olabilir
                    messages = chat.messages
                    for msg in reversed(messages):
                        content = extract_content(msg)
                        if content:
                            assistant_message = content
                            break
                    
                    if not assistant_message:
                        assistant_message = "Yanıt oluşturulamadı."
                else:
                    # Chat objesinin yapısını incele
                    chat_attrs = [attr for attr in dir(chat) if not attr.startswith('__')]
                    print(f"[Agent Chat] UYARI: Mesaj toplanamadı, chat attributes: {chat_attrs}")
                    assistant_message = "Yanıt oluşturuldu ancak parse edilemedi. Lütfen tekrar deneyin."
            except Exception as e:
                assistant_message = f"Yanıt alınırken hata: {str(e)}"
                print(f"[Agent Chat] HATA: Yanıt parse edilemedi: {e}")
                import traceback
                print(traceback.format_exc())
        
        # String'e çevir (güvenlik için) - TextData objelerini de handle et
        if assistant_message:
            if not isinstance(assistant_message, str):
                # TextData veya diğer objeler için extract_content kullan
                extracted = extract_content(assistant_message)
                if extracted:
                    assistant_message = extracted
                else:
                    assistant_message = str(assistant_message)
            
            # Eğer hala TextData string representation'ı varsa parse et
            if isinstance(assistant_message, str) and 'TextData' in assistant_message and 'text=' in assistant_message:
                import re
                # TextData(text="...") formatından text'i çıkar
                match = re.search(r'text=["\']([^"\']+)["\']', assistant_message)
                if match:
                    assistant_message = match.group(1)
                    print(f"[Agent Chat] TextData string'den parse edildi: {len(assistant_message)} karakter")
            
            # Yanıtı temizle ve formatla - gereksiz boşlukları kaldır
            assistant_message = assistant_message.strip()
            
            # JSON formatındaki gereksiz kaçış karakterlerini temizle
            if assistant_message.startswith('"') and assistant_message.endswith('"'):
                try:
                    # json modülü zaten import edilmiş, direkt kullan
                    parsed = json.loads(assistant_message)
                    if isinstance(parsed, str):
                        assistant_message = parsed
                except:
                    pass
            
            # \n karakterlerini düzgün satırlara çevir
            lines = assistant_message.split('\n')
            cleaned_lines = [line.strip() for line in lines if line.strip()]
            assistant_message = '\n'.join(cleaned_lines)
        
        # Eğer hala yanıt yoksa varsayılan mesaj
        if not assistant_message or len(assistant_message.strip()) == 0:
            assistant_message = "Yanıt alınamadı. Lütfen tekrar deneyin."
        
        # Son güvenlik kontrolü - kesinlikle string olduğundan emin ol
        assistant_message = str(assistant_message) if assistant_message else "Yanıt alınamadı."
        
        print(f"[Agent Chat] Final yanıt uzunluğu: {len(assistant_message)} karakter")
        print(f"[Agent Chat] Final yanıt tipi: {type(assistant_message).__name__}")
        print(f"[Agent Chat] Final yanıt preview: {assistant_message[:100]}...")
        
        # JSON serialization için string olduğundan emin ol
        try:
            # Son kontrol: assistant_message kesinlikle string olmalı
            if not isinstance(assistant_message, str):
                assistant_message = str(assistant_message)
            
            response_data = {
                'success': True,
                'response': assistant_message,
                'tool_calls': int(tool_call_count)
            }
            # JSON serialization test (json modülü zaten import edilmiş)
            _ = json.dumps(response_data)  # Test için
            return jsonify(response_data)
        except Exception as json_error:
            print(f"[Agent Chat] JSON serialization hatası: {json_error}")
            import traceback
            print(traceback.format_exc())
            # Fallback: Basit string response
            return jsonify({
                'success': False,
                'error': f'Yanıt serialize edilemedi: {str(json_error)}',
                'response': "Yanıt alındı ancak formatlanamadı. Lütfen konsolu kontrol edin.",
                'tool_calls': 0
            })
        
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"[Agent Chat] HATA: {str(e)}")
        print(f"[Agent Chat] Traceback: {error_trace}")
        return jsonify({
            'error': f'Agent hatası: {str(e)}',
            'details': error_trace if app.debug else None
        }), 500

@app.route('/api/agent/health', methods=['GET'])
def api_agent_health():
    """Agent durumunu kontrol et"""
    try:
        model = get_lm_model()
        is_available = model is not None
        
        return jsonify({
            'available': is_available,
            'tools_count': len(AGENT_TOOLS),
            'model_name': 'qwen/qwen3-vl-4b',
            'supports_images': True,
            'message': 'Model hazır' if is_available else 'LM Studio çalışmıyor veya model yüklü değil. LM Studio\'yu açın ve "qwen/qwen3-vl-4b" modelini yükleyin.'
        })
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"[Health Check] HATA: {str(e)}")
        print(f"[Health Check] Traceback: {error_trace}")
        return jsonify({
            'available': False,
            'tools_count': len(AGENT_TOOLS),
            'model_name': 'qwen/qwen3-vl-4b',
            'supports_images': True,
            'error': str(e),
            'message': f'Health check hatası: {str(e)}'
        }), 500

@app.route('/api/agent/analyze_image', methods=['POST'])
def api_analyze_image():
    """Görsel analiz endpoint'i - dosya veya base64 görsel kabul eder"""
    try:
        question = request.form.get('question', '') or (request.json.get('question', '') if request.is_json else '')
        filepath = None
        
        # Base64 görsel kontrolü (harita screenshot için)
        if request.is_json and 'image_base64' in request.json:
            try:
                base64_data = request.json['image_base64']
                # data:image/png;base64, prefix'ini kaldır
                if ',' in base64_data:
                    base64_data = base64_data.split(',')[1]
                
                # Base64'ü decode et ve dosyaya kaydet
                image_bytes = base64.b64decode(base64_data)
                filename = f"map_screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                
                with open(filepath, 'wb') as f:
                    f.write(image_bytes)
                
                print(f"[Image Analysis] Base64 görsel alındı: {filename}")
            except Exception as e:
                return jsonify({'error': f'Base64 görsel işlenirken hata: {str(e)}'}), 400
        
        # Dosya yükleme kontrolü
        elif 'image' in request.files:
            file = request.files['image']
            if file.filename == '':
                return jsonify({'error': 'Dosya seçilmedi'}), 400
            
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                print(f"[Image Analysis] Dosya yüklendi: {filename}")
            else:
                return jsonify({'error': 'Geçersiz dosya formatı'}), 400
        else:
            return jsonify({'error': 'Görsel dosyası veya base64 görsel bulunamadı'}), 400
        
        if not filepath or not os.path.exists(filepath):
            return jsonify({'error': 'Görsel dosyası kaydedilemedi'}), 500
        
        try:
            # Agent ile görsel analiz
            model = get_lm_model()
            if model is None:
                return jsonify({
                    'error': 'LM Studio model yüklenemedi. LM Studio çalıştırılıyor mu?'
                }), 503
            
            # Görseli hazırla
            image_handle = lms.prepare_image(filepath)
            
            # Soru belirtilmemişse varsayılan soru
            if not question:
                question = "Bu harita veya coğrafi görseli analiz et. Hangi şehirler, bölgeler veya coğrafi özellikler görünüyor? Detaylı bir açıklama yap."
            
            # Chat oluştur
            chat = lms.Chat("""
Sen bir coğrafi analiz uzmanısın. Harita ve coğrafi görselleri analiz edip, 
hangi şehirler, bölgeler, coğrafi özellikler, marker'lar, çizgiler göründüğünü detaylı bir şekilde açıklarsın.
Türkçe yanıt ver.
""")
            chat.add_user_message(question, images=[image_handle])
            
            # Model yanıtı
            prediction = model.respond(chat)
            
            # Yanıtı extract et
            result_text = ""
            if hasattr(prediction, 'content'):
                content = prediction.content
                if isinstance(content, list):
                    text_parts = []
                    for item in content:
                        if hasattr(item, 'text'):
                            text_parts.append(item.text)
                        elif isinstance(item, str):
                            text_parts.append(item)
                    result_text = '\n'.join(text_parts) if text_parts else str(prediction)
                elif isinstance(content, str):
                    result_text = content
                else:
                    result_text = str(content)
            else:
                result_text = str(prediction)
            
            return jsonify({
                'success': True,
                'response': result_text,
                'analysis': result_text,
                'image_filename': os.path.basename(filepath),
                'image_path': os.path.basename(filepath)
            })
            
        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            print(f"[Image Analysis] HATA: {str(e)}")
            print(f"[Image Analysis] Traceback: {error_trace}")
            return jsonify({
                'error': f'Görsel analiz hatası: {str(e)}'
            }), 500
        finally:
            # Geçici dosyayı sakla (debug için)
            pass
    
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"[Image Analysis] Genel HATA: {str(e)}")
        print(f"[Image Analysis] Traceback: {error_trace}")
        return jsonify({
            'error': f'Genel hata: {str(e)}'
        }), 500

@app.route('/api/map/screenshot', methods=['POST'])
def api_map_screenshot():
    """Harita screenshot'ını alıp VLM ile analiz et"""
    try:
        data = request.json
        if not data or 'image_base64' not in data:
            return jsonify({'error': 'Harita screenshot bulunamadı'}), 400
        
        question = data.get('question', 'Haritada şuan ne görünüyor? Detaylı bir açıklama yap.')
        
        # Base64 görseli işle
        base64_data = data['image_base64']
        if ',' in base64_data:
            base64_data = base64_data.split(',')[1]
        
        # Base64'ü decode et ve dosyaya kaydet
        image_bytes = base64.b64decode(base64_data)
        filename = f"map_screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        with open(filepath, 'wb') as f:
            f.write(image_bytes)
        
        print(f"[Map Screenshot] Harita screenshot alındı: {filename}")
        
        # VLM ile analiz et
        model = get_lm_model()
        if model is None:
            return jsonify({
                'error': 'LM Studio model yüklenemedi. LM Studio çalıştırılıyor mu?'
            }), 503
        
        # Görseli hazırla
        image_handle = lms.prepare_image(filepath)
        
        # Chat oluştur
        chat = lms.Chat("""
Sen bir coğrafi analiz uzmanısın. Harita görsellerini analiz edip, 
hangi şehirler, bölgeler, coğrafi özellikler, marker'lar, çizgiler göründüğünü detaylı bir şekilde açıklarsın.
Türkçe yanıt ver.
""")
        chat.add_user_message(question, images=[image_handle])
        
        # Model yanıtı
        prediction = model.respond(chat)
        
        # Yanıtı extract et
        result_text = ""
        if hasattr(prediction, 'content'):
            content = prediction.content
            if isinstance(content, list):
                text_parts = []
                for item in content:
                    if hasattr(item, 'text'):
                        text_parts.append(item.text)
                    elif isinstance(item, str):
                        text_parts.append(item)
                result_text = '\n'.join(text_parts) if text_parts else str(prediction)
            elif isinstance(content, str):
                result_text = content
            else:
                result_text = str(content)
        else:
            result_text = str(prediction)
        
        return jsonify({
            'success': True,
            'analysis': result_text,
            'response': result_text,
            'image_path': filename
        })
        
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"[Map Screenshot] HATA: {str(e)}")
        print(f"[Map Screenshot] Traceback: {error_trace}")
        return jsonify({
            'error': f'Harita screenshot analiz hatası: {str(e)}'
        }), 500

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)

