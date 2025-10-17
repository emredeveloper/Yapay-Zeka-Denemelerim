import torch
import numpy as np
from datasets import load_dataset, Audio
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification, TrainingArguments, Trainer
import re
from collections import Counter
import pandas as pd

# 1. VERİ SETİNİ YÜKLE
print("Veri seti yükleniyor...")
dataset = load_dataset("ysdede/khanacademy-turkish")

# Veriyi incele
print(f"\nVeri seti yapısı: {dataset}")
print(f"\nİlk örnek: {dataset['train'][0]}")
print(f"\nTrain set boyutu: {len(dataset['train'])}")
print(f"Test set boyutu: {len(dataset['test'])}")

# 2. KATEGORİ ETİKETLERİ OLUŞTUR
# Transkripsiyonlardan anahtar kelimelerle kategori çıkar
def categorize_text(text):
    """Metin içeriğinden kategori belirle"""
    text_lower = text.lower()
    
    # Kategori anahtar kelimeleri
    categories = {
        'matematik': ['toplama', 'çıkarma', 'çarpma', 'bölme', 'sayı', 'denklem', 'fonksiyon', 'integral', 'türev', 'geometri'],
        'fizik': ['kuvvet', 'hız', 'ivme', 'newton', 'enerji', 'momentum', 'hareket', 'çekim', 'kutle', 'termodinamik'],
        'kimya': ['atom', 'molekül', 'element', 'bileşik', 'reaksiyon', 'asit', 'baz', 'elektron', 'periyodik'],
        'biyoloji': ['hücre', 'doku', 'organ', 'dna', 'gen', 'protein', 'ekosistem', 'evrim', 'organizm', 'akciğer'],
        'sanat': ['resim', 'müze', 'sanatçı', 'kompozisyon', 'renk', 'tate', 'galeri', 'eser', 'modern', 'rönesans'],
        'ekonomi': ['para', 'banka', 'fiyat', 'piyasa', 'kredi', 'faiz', 'borç', 'maliyet', 'üretim', 'yuan'],
    }
    
    # Her kategori için puan hesapla
    scores = {}
    for category, keywords in categories.items():
        score = sum(1 for keyword in keywords if keyword in text_lower)
        if score > 0:
            scores[category] = score
    
    # En yüksek puanlı kategoriyi döndür
    if scores:
        return max(scores, key=scores.get)
    else:
        return 'genel'  # Varsayılan kategori

# Tüm veri setine kategori ekle
print("\nKategoriler oluşturuluyor...")
dataset = dataset.map(lambda example: {'label': categorize_text(example['transcription'])})

# Kategori dağılımını göster
train_labels = [example['label'] for example in dataset['train']]
label_counts = Counter(train_labels)
print(f"\nKategori dağılımı (train):")
for label, count in sorted(label_counts.items(), key=lambda x: x[1], reverse=True):
    print(f"  {label}: {count} örnek (%{count/len(train_labels)*100:.1f})")

# Label'ları integer'a çevir
unique_labels = sorted(list(set(train_labels)))
label2id = {label: idx for idx, label in enumerate(unique_labels)}
id2label = {idx: label for label, idx in label2id.items()}

print(f"\nLabel mapping: {label2id}")

def add_numeric_label(example):
    example['label_id'] = label2id[example['label']]
    return example

dataset = dataset.map(add_numeric_label)

# 3. SES VERİSİNİ HAZIRLA
print("\nSes verileri işleniyor...")

# Ses örnekleme oranını ayarla (16kHz standart)
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

# 4. MODEL VE FEATURE EXTRACTOR YÜKLE
model_name = "facebook/wav2vec2-base"
print(f"\nModel yükleniyor: {model_name}")

feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
model = AutoModelForAudioClassification.from_pretrained(
    model_name,
    num_labels=len(unique_labels),
    label2id=label2id,
    id2label=id2label,
    ignore_mismatched_sizes=True
)

# 5. ÖN İŞLEME FONKSİYONU
def preprocess_function(examples):
    audio_arrays = [x["array"] for x in examples["audio"]]
    inputs = feature_extractor(
        audio_arrays,
        sampling_rate=feature_extractor.sampling_rate,
        padding=True,
        max_length=16000 * 30,  # Maksimum 30 saniye
        truncation=True,
        return_tensors="pt"
    )
    inputs["labels"] = examples["label_id"]
    return inputs

# Veriyi işle (küçük bir subset ile başla - hız için)
print("\nVeri ön işlemesi yapılıyor...")
train_dataset = dataset['train'].shuffle(seed=42).select(range(min(1000, len(dataset['train']))))
test_dataset = dataset['test'].shuffle(seed=42).select(range(min(200, len(dataset['test']))))

encoded_train = train_dataset.map(
    preprocess_function,
    batched=True,
    batch_size=8,
    remove_columns=train_dataset.column_names
)

encoded_test = test_dataset.map(
    preprocess_function,
    batched=True,
    batch_size=8,
    remove_columns=test_dataset.column_names
)

# 6. METRIK HESAPLAMA
from sklearn.metrics import accuracy_score, f1_score

def compute_metrics(eval_pred):
    predictions = np.argmax(eval_pred.predictions, axis=1)
    accuracy = accuracy_score(eval_pred.label_ids, predictions)
    f1 = f1_score(eval_pred.label_ids, predictions, average='weighted')
    return {"accuracy": accuracy, "f1": f1}

# 7. EĞİTİM AYARLARI
training_args = TrainingArguments(
    output_dir="./turkish-audio-classifier",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    push_to_hub=False,
    logging_steps=10,
    fp16=torch.cuda.is_available(),  # GPU varsa kullan
)

# 8. TRAINER OLUŞTUR
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_train,
    eval_dataset=encoded_test,
    compute_metrics=compute_metrics,
)

# 9. MODELİ EĞİT
print("\n" + "="*50)
print("MODEL EĞİTİMİ BAŞLIYOR")
print("="*50)

trainer.train()

# 10. DEĞERLENDİR
print("\n" + "="*50)
print("MODEL DEĞERLENDİRMESİ")
print("="*50)

results = trainer.evaluate()
print(f"\nTest sonuçları:")
for key, value in results.items():
    print(f"  {key}: {value:.4f}")

# 11. ÖRNEK TAHMİN
print("\n" + "="*50)
print("ÖRNEK TAHMİNLER")
print("="*50)

for i in range(min(5, len(dataset['test']))):
    example = dataset['test'][i]
    
    # Ses verisini işle
    inputs = feature_extractor(
        example["audio"]["array"],
        sampling_rate=16000,
        return_tensors="pt"
    )
    
    # Tahmin yap
    with torch.no_grad():
        logits = model(**inputs).logits
    
    predicted_id = torch.argmax(logits, dim=-1).item()
    predicted_label = id2label[predicted_id]
    actual_label = example['label']
    
    print(f"\nÖrnek {i+1}:")
    print(f"  Metin: {example['transcription'][:100]}...")
    print(f"  Gerçek: {actual_label}")
    print(f"  Tahmin: {predicted_label}")
    print(f"  Doğru: {'✓' if actual_label == predicted_label else '✗'}")

# 12. MODELİ KAYDET
print("\n" + "="*50)
print("MODEL KAYDEDİLİYOR")
print("="*50)

model.save_pretrained("./turkish-audio-classifier-final")
feature_extractor.save_pretrained("./turkish-audio-classifier-final")

print("\n✓ Model başarıyla kaydedildi: ./turkish-audio-classifier-final")
print(f"✓ Toplam {len(unique_labels)} kategori: {', '.join(unique_labels)}")

