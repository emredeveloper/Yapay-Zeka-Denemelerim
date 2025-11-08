import geopandas as gpd
from shapely.geometry import Point
import folium
import requests

# 1. Türkiye il sınırlarını GeoJSON olarak al
# Alternatif URL'ler deneyelim
urls = [
    "https://raw.githubusercontent.com/cihadturhan/tr-geojson/master/geo/tr-cities-utf8.json",
    "https://raw.githubusercontent.com/cihadturhan/tr-geojson/master/geo/tr.json",
    "https://raw.githubusercontent.com/cihadturhan/tr-geojson/master/geo/turkey.json"
]

turkey = None
for url in urls:
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        # GeoJSON verisini parse et
        geojson_data = response.json()
        
        # GeoDataFrame oluştur
        turkey = gpd.GeoDataFrame.from_features(geojson_data.get('features', geojson_data if isinstance(geojson_data, list) else []), crs="EPSG:4326")
        
        if not turkey.empty:
            print(f"GeoJSON başarıyla yüklendi: {url}")
            break
    except Exception as e:
        print(f"URL başarısız ({url}): {e}")
        continue

# Eğer hiçbiri çalışmazsa, geçici dosya yöntemi ile deneyelim
if turkey is None or turkey.empty:
    # Son bir deneme: farklı bir kaynak
    alt_url = "https://raw.githubusercontent.com/deldersveld/topojson/master/countries/turkey/turkey-provinces.json"
    try:
        response = requests.get(alt_url, timeout=10)
        response.raise_for_status()
        geojson_data = response.json()
        turkey = gpd.GeoDataFrame.from_features(geojson_data.get('features', []), crs="EPSG:4326")
        print(f"Alternatif kaynak kullanıldı: {alt_url}")
    except:
        # Eğer hala çalışmazsa, basit bir örnek oluştur
        print("Uyarı: GeoJSON kaynağı bulunamadı. Basit bir harita oluşturuluyor...")
        # En azından şehir noktalarını gösterebilmek için boş bir GeoDataFrame oluştur
        turkey = gpd.GeoDataFrame(columns=['NAME_1', 'geometry'], crs="EPSG:4326")

# 2. Şehir noktalarını oluştur
cities = {
    "Istanbul": (41.0082, 28.9784),
    "Ankara": (39.9208, 32.8541),
    "Izmir": (38.4192, 27.1287),
    "Antalya": (36.8969, 30.7133)
}

geometry = [Point(lon, lat) for lat, lon in cities.values()]
gdf_cities = gpd.GeoDataFrame({'City': list(cities.keys())}, geometry=geometry, crs="EPSG:4326")

# 3. Noktaların hangi il sınırında olduğunu bul
if turkey is not None and not turkey.empty:
    try:
        # NAME_1 kolonu yoksa alternatif kolon adlarını dene
        name_col = None
        for col in ['NAME_1', 'name', 'NAME', 'name_1', 'il', 'city']:
            if col in turkey.columns:
                name_col = col
                break
        
        if name_col:
            joined = gpd.sjoin(gdf_cities, turkey, how="left", predicate="within")
            # Sonuçları yazdır
            if name_col in joined.columns:
                print(joined[["City", name_col]])
            else:
                print(joined[["City"]])
        else:
            joined = gpd.sjoin(gdf_cities, turkey, how="left", predicate="within")
            print("Şehir noktaları:")
            print(joined[["City"]])
    except Exception as e:
        print(f"Spatial join hatası: {e}")
        joined = gdf_cities
else:
    print("GeoJSON verisi bulunamadı, sadece şehir noktaları gösterilecek.")
    joined = gdf_cities

# 5. Harita oluştur
m = folium.Map(location=[39.0, 35.0], zoom_start=6)

# İl sınırlarını ekle (eğer varsa)
if turkey is not None and not turkey.empty:
    try:
        folium.GeoJson(turkey, name="Türkiye İl Sınırları").add_to(m)
    except Exception as e:
        print(f"İl sınırları eklenirken hata: {e}")

# Noktaları ekle
for city, (lat, lon) in cities.items():
    folium.Marker(location=[lat, lon], popup=city).add_to(m)

m.save("turkey_with_cities.html")
print("Harita oluşturuldu: turkey_with_cities.html")