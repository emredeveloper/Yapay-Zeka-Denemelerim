import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from rapidfuzz import fuzz, distance
import jellyfish
import textdistance

# Görselleştirme ayarları
sns.set_style("whitegrid")
plt.rcParams['figure.facecolor'] = 'none'
plt.rcParams['axes.facecolor'] = 'none'

# Sayfa düzeni
st.set_page_config(
    page_title="Metin Benzerlik Analizi",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Örnek veri seti
example_pairs = [
    ("kitten", "sitting"),
    ("flaw", "lawn"),
    ("merhaba dünya", "merhaba dünya"),
    ("python", "piton"),
    ("Türkiye", "Türkiye Cumhuriyeti")
]

def calculate_similarities(text1: str, text2: str) -> dict:
    """Tüm benzerlik metriklerini hızlı kütüphanelerle hesaplar"""
    return {
        "Levenshtein": fuzz.ratio(text1, text2),
        "Jaro-Winkler": fuzz.WRatio(text1, text2),  # Ağırlıklı oran (Jaro-Winkler tabanlı)
        "Jaccard": textdistance.jaccard.normalized_similarity(text1, text2) * 100,
        "Damerau-Levenshtein": (1 - (distance.DamerauLevenshtein.normalized_distance(text1, text2))) * 100,
        "Jaro": jellyfish.jaro_similarity(text1, text2) * 100  # Jellyfish ile Jaro benzerliği
    }

def plot_similarity_scores(scores: dict):
    """Benzerlik skorlarını daha temiz bir grafikle göster"""
    df = pd.DataFrame(scores.items(), columns=['Metrik', 'Skor'])
    
    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.barh(df['Metrik'], df['Skor'], color='#4CAF50')
    
    # Çubuk değerlerini ekle
    for bar in bars:
        width = bar.get_width()
        ax.text(min(width + 2, 95),  # Metin çubuğun dışına taşmasın
                bar.get_y() + bar.get_height()/2,
                f'{width:.1f}%',
                va='center',
                color='black' if width < 90 else 'white')
    
    # Grafik düzenlemeleri
    ax.set_xlim(0, 105)
    ax.set_xlabel('Benzerlik Yüzdesi (%)')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    return fig

def main():
    st.title("🔍 Metin Benzerlik Karşılaştırma")
    st.caption("İki metin arasındaki benzerliği farklı algoritmalarla ölçün")
    
    # Yan çubuk
    with st.sidebar:
        st.subheader("🔧 Ayarlar")
        
        # Hızlı örnek seçimi
        example_select = st.selectbox(
            "Hızlı Örnekler:",
            ["Seçiniz..."] + [f"{p[0]} ↔ {p[1]}" for p in example_pairs],
            index=0
        )
        
        # Örnek metinleri ayır ve varsayılan değerleri ayarla
        default_text1 = ""
        default_text2 = ""
        
        if example_select != "Seçiniz...":
            example_idx = [i for i, p in enumerate(example_pairs) 
                          if f"{p[0]} ↔ {p[1]}" == example_select][0]
            default_text1, default_text2 = example_pairs[example_idx]
        
        # Metin giriş alanları
        text1 = st.text_area("İlk Metin", value=default_text1, height=100)
        text2 = st.text_area("İkinci Metin", value=default_text2, height=100)
        
        st.markdown("---")
        st.caption("ℹ️ Hesaplanan Metrikler:")
        st.caption("• Levenshtein: Karakter değişim mesafesi")
        st.caption("• Jaro-Winkler: Ön ek ağırlıklı benzerlik")
        st.caption("• Jaccard: Kelime kümesi benzerliği")
    
    # Benzerlik hesaplamaları
    if st.button("✅ Benzerliği Hesapla", use_container_width=True):
        if not text1.strip() or not text2.strip():
            st.warning("Lütfen her iki metin alanını da doldurun.")
            return
            
        with st.spinner("⏳ Hesaplanıyor..."):
            scores = calculate_similarities(text1, text2)
            
            # Ana sonuç kartı
            avg_score = sum(scores.values()) / len(scores)
            
            # Başlık çubuğu
            st.markdown(f"### 📊 Benzerlik Sonuçları: {avg_score:.1f}%")
            
            # İki sütunlu düzen
            col1, col2 = st.columns([1, 2])
            
            with col1:
                # Metrik kartları
                st.markdown("#### 📈 Metrikler")
                for metric, score in scores.items():
                    st.metric(label=metric, value=f"{score:.1f}%")
                
                # Genel değerlendirme
                st.markdown("#### 🎯 Değerlendirme")
                if avg_score > 80:
                    st.success("🔍 Çok yüksek benzerlik")
                elif avg_score > 60:
                    st.info("🔍 Orta düzey benzerlik")
                else:
                    st.warning("🔍 Düşük benzerlik")
            
            with col2:
                # Grafik gösterimi
                st.markdown("#### 📊 Benzerlik Dağılımı")
                fig = plot_similarity_scores(scores)
                st.pyplot(fig, use_container_width=True)
            
            # Hızlı analiz
            st.markdown("#### 🔍 Hızlı Analiz")
            
            # Ortak kelimeler
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())
            common_words = words1.intersection(words2)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Ortak Kelime Sayısı", len(common_words))
                if common_words:
                    st.caption(", ".join(f'`{w}`' for w in common_words))
                
            with col2:
                st.metric("Karakter Farkı", 
                         abs(len(text1) - len(text2)),
                         delta=f"{len(text1)} ↔ {len(text2)} karakter")
            
            # Metin önizlemeleri
            with st.expander("📝 Metin Karşılaştırması", expanded=False):
                col1, col2 = st.columns(2)
                with col1:
                    st.text_area("İlk Metin", text1, height=150, disabled=True)
                with col2:
                    st.text_area("İkinci Metin", text2, height=150, disabled=True)

if __name__ == "__main__":
    main()
