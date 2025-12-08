from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# Dengesiz veri seti oluştur (imbalanced dataset)
# %90 sınıf 0, %10 sınıf 1
X, y = make_classification(
    n_samples=1000,
    n_features=20,
    n_informative=15,
    n_redundant=5,
    n_classes=2,
    weights=[0.9, 0.1],  # Dengesiz dağılım
    random_state=42
)

# Veriyi eğitim ve test setlerine ayır
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"Eğitim seti boyutu: {len(X_train)}")
print(f"Test seti boyutu: {len(X_test)}")
print(f"Eğitim setindeki sınıf dağılımı: {np.bincount(y_train)}")
print(f"Test setindeki sınıf dağılımı: {np.bincount(y_test)}")
print("-" * 60)


# Baseline: Standart kNN (uniform weights)
print("\n=== Baseline: Standart kNN (Uniform) ===")
knn_standard = KNeighborsClassifier(n_neighbors=7, weights='uniform')
knn_standard.fit(X_train, y_train)
predictions_standard = knn_standard.predict(X_test)
print("\nSınıflandırma Raporu:")
print(classification_report(y_test, predictions_standard))
print("\nKarmaşıklık Matrisi:")
print(confusion_matrix(y_test, predictions_standard))


# Çözüm 1: Mesafe-Ağırlıklı kNN
print("\n=== Çözüm 1: Mesafe-Ağırlıklı kNN ===")
knn_weighted = KNeighborsClassifier(n_neighbors=7, weights='distance')
knn_weighted.fit(X_train, y_train)
predictions = knn_weighted.predict(X_test)
print("\nSınıflandırma Raporu:")
print(classification_report(y_test, predictions))
print("\nKarmaşıklık Matrisi:")
print(confusion_matrix(y_test, predictions))


# Çözüm 2: Dinamik k Güncellemesi
class DynamicKNN:
    def __init__(self, k=7):
        self.k = k
        self.knn = KNeighborsClassifier(n_neighbors=k)
    
    def fit(self, X, y):
        self.knn.fit(X, y)
        self.y_train = y
        return self
    
    def predict(self, X):
        # k en yakın komşuyu bul
        distances, indices = self.knn.kneighbors(X)
        predictions = []
        
        for neighbor_indices in indices:
            # Her sınıfın sayısını bul
            neighbors = self.y_train[neighbor_indices]
            class_counts = np.bincount(neighbors)
            
            # Dinamik k' hesapla (en az temsil edilen sınıfın sayısı)
            # Eğer bir sınıf hiç yoksa, en az 1 komşu kullan
            k_prime = max(1, np.min(class_counts[class_counts > 0]) if np.any(class_counts > 0) else 1)
            
            # Eğer tüm komşular aynı sınıftansa, çoğunluk oylaması yap
            if len(np.unique(neighbors)) == 1:
                prediction = neighbors[0]
            else:
                # k' komşu arasında oylama yap
                top_neighbors = neighbors[:k_prime]
                prediction = np.bincount(top_neighbors).argmax()
            
            predictions.append(prediction)
        
        return np.array(predictions)

# Kullanım
print("\n=== Çözüm 2: Dinamik k Güncellemesi ===")
dynamic_knn = DynamicKNN(k=7)
dynamic_knn.fit(X_train, y_train)
predictions_dynamic = dynamic_knn.predict(X_test)
print("\nSınıflandırma Raporu:")
print(classification_report(y_test, predictions_dynamic))
print("\nKarmaşıklık Matrisi:")
print(confusion_matrix(y_test, predictions_dynamic))


# Performans Karşılaştırması
print("\n" + "=" * 80)
print("PERFORMANS KARŞILAŞTIRMASI - ÖZET")
print("=" * 80)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Metrikleri hesapla
models = {
    'Standart kNN (Uniform)': predictions_standard,
    'Mesafe-Ağırlıklı kNN': predictions,
    'Dinamik k kNN': predictions_dynamic
}

print(f"\n{'Model':<30} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
print("-" * 80)

for model_name, preds in models.items():
    acc = accuracy_score(y_test, preds)
    # Azınlık sınıfı (class 1) için metrikler
    prec = precision_score(y_test, preds, pos_label=1, zero_division=0)
    rec = recall_score(y_test, preds, pos_label=1)
    f1 = f1_score(y_test, preds, pos_label=1)
    
    print(f"{model_name:<30} {acc:<12.4f} {prec:<12.4f} {rec:<12.4f} {f1:<12.4f}")

print("\n* Precision, Recall ve F1-Score metrikleri azınlık sınıfı (class 1) için hesaplanmıştır.")
print("* Dengesiz veri setlerinde azınlık sınıfının doğru tahmin edilmesi kritik öneme sahiptir.")
print("=" * 80)