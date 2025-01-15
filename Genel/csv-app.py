import sys
import pandas as pd
from PyQt6.QtWidgets import (
    QApplication, QWidget, QLabel, QVBoxLayout, QComboBox, QPushButton, QFileDialog
)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

class CSVAnalyzer(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CSV Veri Analiz ve Model Önerisi")
        self.layout = QVBoxLayout()
        
        # CSV dosyası yükleme butonu
        self.loadButton = QPushButton("CSV Dosyası Yükle")
        self.loadButton.clicked.connect(self.load_csv)
        self.layout.addWidget(self.loadButton)
        
        # Sütunlar için ComboBox'lar
        self.columnXLabel = QLabel("X Sütunu:")
        self.columnXSelect = QComboBox()
        self.layout.addWidget(self.columnXLabel)
        self.layout.addWidget(self.columnXSelect)
        
        self.columnYLabel = QLabel("Y Sütunu:")
        self.columnYSelect = QComboBox()
        self.layout.addWidget(self.columnYLabel)
        self.layout.addWidget(self.columnYSelect)
        
        # Dağılım Gösterme Butonu
        self.showDistributionButton = QPushButton("Dağılımı Göster")
        self.showDistributionButton.clicked.connect(self.show_distribution)
        self.layout.addWidget(self.showDistributionButton)
        
        # Model önerisi butonu
        self.showModelButton = QPushButton("En Uygun Modeli Göster")
        self.showModelButton.clicked.connect(self.suggest_model)
        self.layout.addWidget(self.showModelButton)
        
        # Sonuçları ekranda göster
        self.resultLabel = QLabel("")
        self.layout.addWidget(self.resultLabel)
        
        self.setLayout(self.layout)
        self.data = None

    def load_csv(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "CSV Dosyası Yükle", "", "CSV Dosyaları (*.csv)")
        if file_path:
            self.data = pd.read_csv(file_path)
            self.columnXSelect.clear()
            self.columnYSelect.clear()
            self.columnXSelect.addItems(self.data.columns)
            self.columnYSelect.addItems(self.data.columns)
    
    def show_distribution(self):
        if self.data is None:
            self.resultLabel.setText("Lütfen önce bir CSV dosyası yükleyin.")
            return
        
        x_column = self.columnXSelect.currentText()
        y_column = self.columnYSelect.currentText()
        
        if x_column and y_column:
            plt.figure(figsize=(10, 6))
            plt.scatter(self.data[x_column], self.data[y_column])
            plt.xlabel(x_column)
            plt.ylabel(y_column)
            plt.title(f"{x_column} ve {y_column} Dağılımı")
            plt.show()
    
    def suggest_model(self):
        if self.data is None:
            self.resultLabel.setText("Lütfen önce bir CSV dosyası yükleyin.")
            return
        
        x_column = self.columnXSelect.currentText()
        y_column = self.columnYSelect.currentText()
        
        if x_column and y_column:
            X = self.data[[x_column]].copy()
            y = self.data[y_column]
            
            # Eğer X sütunu tarih formatındaysa datetime'a çevir
            if not pd.api.types.is_datetime64_any_dtype(X[x_column]):
                try:
                    X[x_column] = pd.to_datetime(X[x_column], errors='coerce')
                    # Tarih formatını sayısal değerlere dönüştür
                    X[x_column] = X[x_column].map(pd.Timestamp.timestamp)
                except Exception as e:
                    self.resultLabel.setText(f"Tarih dönüşümü hatası: {str(e)}")
                    return
            
            if y.dtype == 'object':
                le = LabelEncoder()
                y = le.fit_transform(y)
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Modelleri eğit ve karşılaştır
            models = {
                'Lojistik Regresyon': LogisticRegression(),
                'Rastgele Orman': RandomForestClassifier(),
                'Destek Vektör Makinesi': SVC()
            }
            best_model = None
            best_score = 0
            
            for name, model in models.items():
                model.fit(X_train, y_train)
                predictions = model.predict(X_test)
                score = accuracy_score(y_test, predictions)
                if score > best_score:
                    best_score = score
                    best_model = name
            
            self.resultLabel.setText(f"En uygun model: {best_model} (Doğruluk: {best_score:.2f})")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CSVAnalyzer()
    window.show()
    sys.exit(app.exec())