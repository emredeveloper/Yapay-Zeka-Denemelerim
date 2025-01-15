import csv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
import string
from TurkishStemmer import TurkishStemmer

stemmer = TurkishStemmer()

def preprocess_text(text):
    # Küçük harfe çevir
    text = text.lower()
    # Noktalama işaretlerini kaldır
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Fazla boşlukları kaldır
    text = re.sub(r'\s+', ' ', text).strip()
    # Kelimeleri köklerine ayır
    words = text.split()
    stemmed_words = [stemmer.stem(word) for word in words]
    return ' '.join(stemmed_words)

def load_csv(file_path):
    timestamps = []
    texts = []
    original_texts = []
    with open(file_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Başlık satırını atla
        for row in reader:
            timestamps.append(row[0])
            original_texts.append(row[1])
            texts.append(preprocess_text(row[1]))
    return timestamps, texts, original_texts

def find_most_similar(query, texts, timestamps, original_texts, top_n=3):
    # TF-IDF vektörleştiriciyi oluştur
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))  # Unigram ve bigram kullan
    
    # Metinleri ve sorguyu vektörleştir
    tfidf_matrix = vectorizer.fit_transform(texts + [preprocess_text(query)])
    
    # Kosinüs benzerliğini hesapla
    cosine_similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1]).flatten()
    
    # Sıfır olmayan benzerlik skorlarını filtrele
    non_zero_indices = np.where(cosine_similarities > 0)[0]
    
    if len(non_zero_indices) == 0:
        return []
    
    # En benzer metinlerin indekslerini bul
    most_similar_indices = non_zero_indices[np.argsort(cosine_similarities[non_zero_indices])[-top_n:][::-1]]
    
    # Sonuçları döndür
    results = []
    for idx in most_similar_indices:
        results.append({
            'timestamp': timestamps[idx],
            'text': original_texts[idx],
            'processed_text': texts[idx],
            'similarity': cosine_similarities[idx]
        })
    
    return results

# CSV dosyasını yükle
csv_file = "ses_metinleri_ve_zaman_damgalari.csv"
timestamps, texts, original_texts = load_csv(csv_file)

# Kullanıcıdan sorgu al
query = input("Lütfen bir sorgu girin: ")

# En benzer metinleri bul
similar_texts = find_most_similar(query, texts, timestamps, original_texts)

# Sonuçları yazdır
if similar_texts:
    print("\nEn benzer metinler:")
    for result in similar_texts:
        print(f"Zaman: {result['timestamp']}")
        print(f"Orijinal Metin: {result['text']}")
        print(f"İşlenmiş Metin: {result['processed_text']}")
        print(f"Benzerlik: {result['similarity']:.4f}")
        print()
else:
    print("\nHiçbir benzer metin bulunamadı.")