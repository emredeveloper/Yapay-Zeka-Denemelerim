import geopandas as gpd
from shapely.geometry import Point
import folium
from folium.plugins import MarkerCluster, MeasureControl, Draw
import requests
import gradio as gr
import pandas as pd
from math import radians, cos, sin, asin, sqrt
import json
import random
from datetime import datetime

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
        # Şehir adını Wikipedia formatına çevir
        wiki_name = TURKIYE_SEHIRLER.get(city_name, {}).get("wikipedia", city_name)
        
        # Önce Türkçe deneyelim
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
            # İngilizce deneyelim
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
    except Exception as e:
        pass
    
    return {
        "ozet": f"{city_name} hakkında Wikipedia bilgisi alınamadı.",
        "baslik": city_name,
        "url": "",
        "resim": ""
    }

def get_city_info(city_name):
    """Şehir hakkında detaylı bilgi (sentetik veri + Wikipedia)"""
    if city_name not in TURKIYE_SEHIRLER:
        return "Şehir bulunamadı!"
    
    city_data = TURKIYE_SEHIRLER[city_name]
    lat, lon = city_data["koordinat"]
    
    # Wikipedia bilgisi
    wiki_info = get_wikipedia_info(city_name)
    
    # Bilgi formatla
    info = f"""
# {city_name} Şehir Bilgileri

## 📍 Coğrafi Bilgiler
- **Koordinatlar**: {lat:.4f}°N, {lon:.4f}°E
- **Bölge**: {city_data['bolge']}
- **Yükseklik**: {city_data['yukseklik_m']} metre
- **Alan**: {city_data['alan_km2']:,} km²
- **Plaka Kodu**: {city_data['plaka']:02d}

## 👥 Demografik Bilgiler
- **Nüfus**: {city_data['nufus']:,}
- **Nüfus Yoğunluğu**: {city_data['nufus']/city_data['alan_km2']:.2f} kişi/km²

## 📚 Wikipedia Bilgisi
{wiki_info['ozet']}

[Wikipedia'da Daha Fazla Bilgi]({wiki_info['url']})
    """
    
    return info.strip()

def get_cities_table(selected_cities):
    """Seçilen şehirlerin tablosu"""
    if not selected_cities:
        return pd.DataFrame()
    
    cities_list = [city for city in selected_cities if city in TURKIYE_SEHIRLER]
    
    table_data = []
    for city in cities_list:
        data = TURKIYE_SEHIRLER[city]
        table_data.append({
            "Şehir": city,
            "Bölge": data["bolge"],
            "Nüfus": f"{data['nufus']:,}",
            "Alan (km²)": f"{data['alan_km2']:,}",
            "Yükseklik (m)": data["yukseklik_m"],
            "Plaka": f"{data['plaka']:02d}"
        })
    
    return pd.DataFrame(table_data)

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
    """İki nokta arasındaki mesafeyi km cinsinden hesapla (Haversine formülü)"""
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371  # Dünya yarıçapı (km)
    return c * r

def create_map(selected_cities, show_boundaries, show_clustering, map_style, show_lines=True):
    """Harita oluştur"""
    if not selected_cities:
        return None, "Lütfen en az bir şehir seçin!"
    
    # Seçilen şehirleri filtrele
    cities_to_show = {city: TURKIYE_SEHIRLER[city] for city in selected_cities if city in TURKIYE_SEHIRLER}
    
    if not cities_to_show:
        return None, "Geçerli şehir seçilmedi!"
    
    # Harita tile seçimi
    tile_options = {
        "OpenStreetMap": "OpenStreetMap",
        "Satellite": "Esri.WorldImagery",
        "Terrain": "OpenTopoMap",
        "Dark": "CartoDB.DarkMatter"
    }
    tiles = tile_options.get(map_style, "OpenStreetMap")
    
    # Harita oluştur
    center_lat = sum(data["koordinat"][0] for data in cities_to_show.values()) / len(cities_to_show)
    center_lon = sum(data["koordinat"][1] for data in cities_to_show.values()) / len(cities_to_show)
    
    m = folium.Map(location=[center_lat, center_lon], zoom_start=6, tiles=tiles)
    
    # Marker clustering ekle
    if show_clustering and len(cities_to_show) > 1:
        marker_cluster = MarkerCluster().add_to(m)
        marker_group = marker_cluster
    else:
        marker_group = m
    
    # Türkiye il sınırlarını ekle
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
    
    # Şehirleri haritaya ekle (zengin popup'larla)
    for city, city_data in cities_to_show.items():
        lat, lon = city_data["koordinat"]
        
        # Popup içeriği oluştur
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
        
        # Marker rengini nüfusa göre ayarla
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
    
    # Mesafe çizgileri ve bilgileri ekle
    if show_lines and len(cities_to_show) > 1:
        city_list = list(cities_to_show.items())
        for i in range(len(city_list)):
            for j in range(i + 1, len(city_list)):
                city1, data1 = city_list[i]
                city2, data2 = city_list[j]
                lat1, lon1 = data1["koordinat"]
                lat2, lon2 = data2["koordinat"]
                distance = haversine(lon1, lat1, lon2, lat2)
                
                # Çizgi ekle
                folium.PolyLine(
                    locations=[[lat1, lon1], [lat2, lon2]],
                    color='blue',
                    weight=2,
                    opacity=0.5,
                    popup=f"<b>{city1} ↔ {city2}</b><br>Mesafe: {distance:.2f} km"
                ).add_to(m)
    
    # Katman kontrolü ekle
    folium.LayerControl().add_to(m)
    
    # Ölçüm aracı ekle
    MeasureControl().add_to(m)
    
    # Çizim aracı ekle
    Draw(export=True).add_to(m)
    
    # HTML olarak kaydet
    map_html = m._repr_html_()
    return map_html, f"✅ Harita başarıyla oluşturuldu! {len(cities_to_show)} şehir gösteriliyor."

def calculate_distances(selected_cities):
    """Şehirler arası mesafeleri hesapla"""
    if len(selected_cities) < 2:
        return "En az 2 şehir seçmelisiniz!"
    
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
                "Şehir 1": city1,
                "Şehir 2": city2,
                "Mesafe (km)": f"{distance:.2f}"
            })
    
    df = pd.DataFrame(distances)
    return df.to_string(index=False)

def get_statistics(selected_cities):
    """İstatistikler hesapla"""
    if not selected_cities:
        return "Lütfen şehir seçin!"
    
    cities_list = [city for city in selected_cities if city in TURKIYE_SEHIRLER]
    
    if len(cities_list) < 2:
        city_data = TURKIYE_SEHIRLER[cities_list[0]]
        lat, lon = city_data["koordinat"]
        return f"""📍 Seçilen Şehir: {cities_list[0]}

📊 Bilgiler:
- Koordinat: ({lat:.4f}, {lon:.4f})
- Nüfus: {city_data['nufus']:,}
- Alan: {city_data['alan_km2']:,} km²
- Bölge: {city_data['bolge']}
"""
    
    # En uzun ve en kısa mesafeleri bul
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
            distances.append((city1, city2, distance))
    
    if distances:
        max_dist = max(distances, key=lambda x: x[2])
        min_dist = min(distances, key=lambda x: x[2])
        avg_dist = sum(d[2] for d in distances) / len(distances)
        
        # Nüfus ve alan istatistikleri
        max_pop_city = max(cities_list, key=lambda c: TURKIYE_SEHIRLER[c]['nufus'])
        min_pop_city = min(cities_list, key=lambda c: TURKIYE_SEHIRLER[c]['nufus'])
        
        stats = f"""
📊 İSTATİSTİKLER

📍 Seçilen Şehir Sayısı: {len(cities_list)}
📏 Toplam Mesafe Çifti: {len(distances)}

🔴 En Uzak Şehirler:
   {max_dist[0]} ↔ {max_dist[1]}: {max_dist[2]:.2f} km

🟢 En Yakın Şehirler:
   {min_dist[0]} ↔ {min_dist[1]}: {min_dist[2]:.2f} km

📈 Ortalama Mesafe: {avg_dist:.2f} km

👥 Demografik İstatistikler:
   Toplam Nüfus: {total_pop:,}
   Ortalama Nüfus: {total_pop/len(cities_list):,.0f}
   En Kalabalık: {max_pop_city} ({TURKIYE_SEHIRLER[max_pop_city]['nufus']:,})
   En Az Nüfuslu: {min_pop_city} ({TURKIYE_SEHIRLER[min_pop_city]['nufus']:,})

🗺️ Coğrafi İstatistikler:
   Toplam Alan: {total_area:,} km²
   Ortalama Alan: {total_area/len(cities_list):,.0f} km²
        """
        return stats
    return "İstatistik hesaplanamadı!"

# Gradio arayüzü
with gr.Blocks(title="Türkiye Coğrafi Analiz") as demo:
    gr.Markdown("# 🗺️ Türkiye Coğrafi Analiz ve Haritalama Uygulaması")
    gr.Markdown("Şehirler arası mesafe hesaplama, haritalama ve analiz yapın!")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### ⚙️ Ayarlar")
            
            selected_cities = gr.CheckboxGroup(
                choices=list(TURKIYE_SEHIRLER.keys()),
                value=["Istanbul", "Ankara", "Izmir"],
                label="📍 Şehirler Seçin",
                interactive=True
            )
            
            show_boundaries = gr.Checkbox(
                value=True,
                label="🗺️ İl Sınırlarını Göster",
                interactive=True
            )
            
            show_clustering = gr.Checkbox(
                value=False,
                label="🔗 Marker Clustering",
                interactive=True
            )
            
            map_style = gr.Radio(
                choices=["OpenStreetMap", "Satellite", "Terrain", "Dark"],
                value="OpenStreetMap",
                label="🎨 Harita Stili",
                interactive=True
            )
            
            show_lines = gr.Checkbox(
                value=True,
                label="📏 Mesafe Çizgilerini Göster",
                interactive=True
            )
            
            create_btn = gr.Button("🗺️ Harita Oluştur", variant="primary", scale=1)
        
        with gr.Column(scale=2):
            gr.Markdown("### 🗺️ Harita")
            map_output = gr.HTML(label="Harita")
            status_output = gr.Textbox(label="Durum", interactive=False)
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 📊 İstatistikler")
            stats_output = gr.Textbox(label="İstatistikler", lines=12, interactive=False)
            stats_btn = gr.Button("📊 İstatistikleri Hesapla")
        
        with gr.Column():
            gr.Markdown("### 📏 Mesafe Tablosu")
            distance_output = gr.Textbox(label="Mesafeler", lines=12, interactive=False)
            distance_btn = gr.Button("📏 Mesafeleri Hesapla")
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 📋 Şehir Tablosu")
            city_table = gr.Dataframe(
                label="Seçilen Şehirlerin Bilgileri",
                interactive=False,
                wrap=True
            )
            table_btn = gr.Button("📋 Tabloyu Güncelle")
        
        with gr.Column():
            gr.Markdown("### 📚 Şehir Detayları (Wikipedia)")
            city_selector = gr.Dropdown(
                choices=list(TURKIYE_SEHIRLER.keys()),
                label="Şehir Seçin",
                value="Istanbul"
            )
            city_info_output = gr.Markdown(label="Şehir Bilgileri")
            city_info_btn = gr.Button("📚 Bilgileri Yükle")
    
    # Event handlers
    create_btn.click(
        fn=create_map,
        inputs=[selected_cities, show_boundaries, show_clustering, map_style, show_lines],
        outputs=[map_output, status_output]
    )
    
    stats_btn.click(
        fn=get_statistics,
        inputs=[selected_cities],
        outputs=[stats_output]
    )
    
    distance_btn.click(
        fn=calculate_distances,
        inputs=[selected_cities],
        outputs=[distance_output]
    )
    
    table_btn.click(
        fn=get_cities_table,
        inputs=[selected_cities],
        outputs=[city_table]
    )
    
    city_info_btn.click(
        fn=get_city_info,
        inputs=[city_selector],
        outputs=[city_info_output]
    )
    
    # Sayfa yüklendiğinde otomatik harita oluştur
    demo.load(
        fn=create_map,
        inputs=[selected_cities, show_boundaries, show_clustering, map_style, show_lines],
        outputs=[map_output, status_output]
    )
    
    # Şehir seçildiğinde otomatik bilgi yükle
    city_selector.change(
        fn=get_city_info,
        inputs=[city_selector],
        outputs=[city_info_output]
    )
    
    # Şehirler değiştiğinde tabloyu güncelle
    selected_cities.change(
        fn=get_cities_table,
        inputs=[selected_cities],
        outputs=[city_table]
    )

if __name__ == "__main__":
    demo.launch(share=False, server_name="127.0.0.1", server_port=7860)

