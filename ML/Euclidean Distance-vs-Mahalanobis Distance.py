import numpy as np
from scipy.spatial.distance import euclidean, mahalanobis
from scipy import stats
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.columns import Columns
from rich.progress import Progress
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

console = Console()

# Ayırıcı çizgi fonksiyonu
def print_separator(char="═", length=70):
    console.print(f"[bold yellow]{char * length}[/bold yellow]")

def print_section_header(title, emoji="📊"):
    console.print()
    print_separator()
    console.print(f"[bold cyan]{emoji} {title}[/bold cyan]")
    print_separator()
    console.print()

# ═══════════════════════════════════════════════════════════════════════════
# VERİ SETİ YÜKLEME
# ═══════════════════════════════════════════════════════════════════════════

console.print(Panel.fit(
    "[bold white on blue] Euclidean vs Mahalanobis Distance - Kapsamlı Analiz [/bold white on blue]", 
    border_style="blue",
    padding=(1, 2)
))

iris = load_iris()
data = iris.data
mean = np.mean(data, axis=0)
feature_names = ['Sepal Uzunluğu', 'Sepal Genişliği', 'Petal Uzunluğu', 'Petal Genişliği']

# Veri seti bilgileri
print_section_header("VERİ SETİ BİLGİLERİ", "📁")

info_table = Table(show_header=False, border_style="blue", box=None)
info_table.add_column("Özellik", style="bold cyan", width=20)
info_table.add_column("Değer", style="white")

info_table.add_row("📌 Veri Seti", "Iris Dataset (Fisher, 1936)")
info_table.add_row("📊 Boyut", f"{data.shape[0]} örnek × {data.shape[1]} özellik")
info_table.add_row("🏷️ Sınıflar", "Setosa, Versicolor, Virginica")
info_table.add_row("📏 Özellikler", ", ".join(iris.feature_names))

console.print(info_table)

# Temel istatistikler
print_section_header("TEMEL İSTATİSTİKLER", "📈")

stats_basic = Table(title="Özellik İstatistikleri", show_header=True, header_style="bold magenta")
stats_basic.add_column("Özellik", style="cyan", width=16)
stats_basic.add_column("Ortalama", justify="right", width=10)
stats_basic.add_column("Std Sapma", justify="right", width=10)
stats_basic.add_column("Min", justify="right", width=8)
stats_basic.add_column("Max", justify="right", width=8)
stats_basic.add_column("Medyan", justify="right", width=8)

for i, name in enumerate(feature_names):
    col_data = data[:, i]
    stats_basic.add_row(
        name,
        f"{np.mean(col_data):.2f}",
        f"{np.std(col_data):.2f}",
        f"{np.min(col_data):.2f}",
        f"{np.max(col_data):.2f}",
        f"{np.median(col_data):.2f}"
    )

console.print(stats_basic)

# Korelasyon matrisi
print_section_header("KORELASYON MATRİSİ", "🔗")

corr_matrix = np.corrcoef(data.T)
corr_table = Table(title="Pearson Korelasyon Katsayıları", show_header=True, header_style="bold magenta")
corr_table.add_column("", style="cyan", width=16)
for name in feature_names:
    corr_table.add_column(name[:8], justify="center", width=10)

for i, row_name in enumerate(feature_names):
    row_values = []
    for j in range(len(feature_names)):
        val = corr_matrix[i, j]
        if val == 1.0:
            color = "white"
        elif val > 0.7:
            color = "green"
        elif val < -0.7:
            color = "red"
        elif val > 0.4:
            color = "yellow"
        else:
            color = "dim"
        row_values.append(f"[{color}]{val:.3f}[/{color}]")
    corr_table.add_row(row_name, *row_values)

console.print(corr_table)
console.print("[dim]💡 Yeşil: Güçlü pozitif korelasyon (>0.7) | Kırmızı: Güçlü negatif (<-0.7)[/dim]")

# ═══════════════════════════════════════════════════════════════════════════
# EUCLIDEAN DISTANCE HESAPLAMA
# ═══════════════════════════════════════════════════════════════════════════

print_section_header("EUCLIDEAN DISTANCE (ÖKLIDYEN MESAFE)", "📐")

console.print(Panel(
    "[cyan]Euclidean mesafe, iki nokta arasındaki düz çizgi mesafesini ölçer.\n"
    "Formül: d(x,y) = √Σ(xᵢ - yᵢ)²[/cyan]\n\n"
    "[yellow]✓ Avantajlar:[/yellow] Basit, sezgisel, hızlı hesaplama\n"
    "[red]✗ Dezavantajlar:[/red] Ölçek bağımlı, korelasyonu göz ardı eder",
    title="[bold]Teorik Bilgi[/bold]",
    border_style="blue"
))

euclidean_distances = np.array([euclidean(point, mean) for point in data])

table_euc = Table(title="🎯 Euclidean - En Uzak 10 Nokta (Potansiyel Outlier'lar)", 
                  show_header=True, header_style="bold magenta",
                  border_style="green")
table_euc.add_column("Sıra", style="bold cyan", justify="center", width=6)
table_euc.add_column("Index", style="dim", justify="center", width=6)
table_euc.add_column("Mesafe", style="green", justify="right", width=10)
table_euc.add_column("Z-Skor", style="yellow", justify="right", width=8)
table_euc.add_column("Sepal L", justify="right", width=8)
table_euc.add_column("Sepal W", justify="right", width=8)
table_euc.add_column("Petal L", justify="right", width=8)
table_euc.add_column("Petal W", justify="right", width=8)
table_euc.add_column("Sınıf", style="magenta", justify="center", width=12)

euc_mean = euclidean_distances.mean()
euc_std = euclidean_distances.std()
species = ['Setosa', 'Versicolor', 'Virginica']

top_10_euc = np.argsort(euclidean_distances)[-10:][::-1]
for rank, idx in enumerate(top_10_euc, 1):
    point = data[idx]
    dist = euclidean_distances[idx]
    z_score = (dist - euc_mean) / euc_std
    table_euc.add_row(
        str(rank),
        str(idx),
        f"{dist:.4f}",
        f"{z_score:.2f}",
        f"{point[0]:.2f}",
        f"{point[1]:.2f}",
        f"{point[2]:.2f}",
        f"{point[3]:.2f}",
        species[iris.target[idx]]
    )

console.print(table_euc)

# ═══════════════════════════════════════════════════════════════════════════
# MAHALANOBIS DISTANCE HESAPLAMA
# ═══════════════════════════════════════════════════════════════════════════

print_section_header("MAHALANOBIS DISTANCE (MAHALANOBİS MESAFE)", "📏")

console.print(Panel(
    "[cyan]Mahalanobis mesafe, kovaryans matrisini kullanarak verilerin\n"
    "dağılım yapısını dikkate alır.\n"
    "Formül: d(x) = √((x-μ)ᵀ Σ⁻¹ (x-μ))[/cyan]\n\n"
    "[yellow]✓ Avantajlar:[/yellow] Ölçekten bağımsız, korelasyonu dikkate alır\n"
    "[red]✗ Dezavantajlar:[/red] Hesaplama karmaşıklığı, tekil matris sorunu",
    title="[bold]Teorik Bilgi[/bold]",
    border_style="blue"
))

cov_matrix = np.cov(data.T)
cov_inv = np.linalg.inv(cov_matrix)

mahal_distances = []
for point in data:
    diff = point - mean
    dist = np.sqrt(diff @ cov_inv @ diff.T)
    mahal_distances.append(dist)

mahal_distances = np.array(mahal_distances)

table_mah = Table(title="🎯 Mahalanobis - En Uzak 10 Nokta (Potansiyel Outlier'lar)", 
                  show_header=True, header_style="bold magenta",
                  border_style="red")
table_mah.add_column("Sıra", style="bold cyan", justify="center", width=6)
table_mah.add_column("Index", style="dim", justify="center", width=6)
table_mah.add_column("Mesafe", style="green", justify="right", width=10)
table_mah.add_column("p-değer", style="yellow", justify="right", width=8)
table_mah.add_column("Sepal L", justify="right", width=8)
table_mah.add_column("Sepal W", justify="right", width=8)
table_mah.add_column("Petal L", justify="right", width=8)
table_mah.add_column("Petal W", justify="right", width=8)
table_mah.add_column("Sınıf", style="magenta", justify="center", width=12)

# Chi-square tabanlı p-değeri (df = 4 özellik için)
df = data.shape[1]
p_values = 1 - stats.chi2.cdf(mahal_distances**2, df)

top_10_mah = np.argsort(mahal_distances)[-10:][::-1]
for rank, idx in enumerate(top_10_mah, 1):
    point = data[idx]
    dist = mahal_distances[idx]
    p_val = p_values[idx]
    p_color = "red" if p_val < 0.05 else "yellow" if p_val < 0.1 else "green"
    table_mah.add_row(
        str(rank),
        str(idx),
        f"{dist:.4f}",
        f"[{p_color}]{p_val:.4f}[/{p_color}]",
        f"{point[0]:.2f}",
        f"{point[1]:.2f}",
        f"{point[2]:.2f}",
        f"{point[3]:.2f}",
        species[iris.target[idx]]
    )

console.print(table_mah)
console.print("[dim]💡 p-değeri < 0.05: İstatistiksel olarak anlamlı outlier (kırmızı)[/dim]")

# ═══════════════════════════════════════════════════════════════════════════
# İSTATİSTİKSEL KARŞILAŞTIRMA
# ═══════════════════════════════════════════════════════════════════════════

print_section_header("İSTATİSTİKSEL KARŞILAŞTIRMA", "📊")

stats_table = Table(title="Mesafe Dağılım İstatistikleri", 
                    show_header=True, header_style="bold magenta",
                    border_style="cyan")
stats_table.add_column("Metrik", style="bold cyan", width=16)
stats_table.add_column("Euclidean", style="yellow", justify="right", width=12)
stats_table.add_column("Mahalanobis", style="yellow", justify="right", width=12)
stats_table.add_column("Fark (%)", style="green", justify="right", width=10)

metrics = [
    ("Ortalama", euclidean_distances.mean(), mahal_distances.mean()),
    ("Std Sapma", euclidean_distances.std(), mahal_distances.std()),
    ("Varyans", euclidean_distances.var(), mahal_distances.var()),
    ("Min", euclidean_distances.min(), mahal_distances.min()),
    ("Max", euclidean_distances.max(), mahal_distances.max()),
    ("Medyan", np.median(euclidean_distances), np.median(mahal_distances)),
    ("IQR", np.percentile(euclidean_distances, 75) - np.percentile(euclidean_distances, 25),
           np.percentile(mahal_distances, 75) - np.percentile(mahal_distances, 25)),
    ("Çarpıklık", stats.skew(euclidean_distances), stats.skew(mahal_distances)),
    ("Basıklık", stats.kurtosis(euclidean_distances), stats.kurtosis(mahal_distances)),
]

for name, euc_val, mah_val in metrics:
    if euc_val != 0:
        diff_pct = ((mah_val - euc_val) / euc_val) * 100
    else:
        diff_pct = 0
    stats_table.add_row(name, f"{euc_val:.4f}", f"{mah_val:.4f}", f"{diff_pct:+.1f}%")

console.print(stats_table)

# ═══════════════════════════════════════════════════════════════════════════
# OUTLIER TESPİT DEĞERLENDİRMESİ
# ═══════════════════════════════════════════════════════════════════════════

print_section_header("OUTLIER TESPİT DEĞERLENDİRMESİ", "🔍")

# Farklı eşik değerleri için outlier sayıları
thresholds = {
    "1.5 IQR": {
        "euclidean": np.sum(euclidean_distances > (np.percentile(euclidean_distances, 75) + 1.5 * (np.percentile(euclidean_distances, 75) - np.percentile(euclidean_distances, 25)))),
        "mahalanobis": np.sum(mahal_distances > (np.percentile(mahal_distances, 75) + 1.5 * (np.percentile(mahal_distances, 75) - np.percentile(mahal_distances, 25))))
    },
    "2 Std Sapma": {
        "euclidean": np.sum(euclidean_distances > euclidean_distances.mean() + 2 * euclidean_distances.std()),
        "mahalanobis": np.sum(mahal_distances > mahal_distances.mean() + 2 * mahal_distances.std())
    },
    "3 Std Sapma": {
        "euclidean": np.sum(euclidean_distances > euclidean_distances.mean() + 3 * euclidean_distances.std()),
        "mahalanobis": np.sum(mahal_distances > mahal_distances.mean() + 3 * mahal_distances.std())
    },
    "Chi² (p<0.05)": {
        "euclidean": "N/A",
        "mahalanobis": np.sum(p_values < 0.05)
    },
    "Chi² (p<0.01)": {
        "euclidean": "N/A",
        "mahalanobis": np.sum(p_values < 0.01)
    }
}

outlier_table = Table(title="Farklı Eşik Değerleri ile Tespit Edilen Outlier Sayıları",
                      show_header=True, header_style="bold magenta",
                      border_style="yellow")
outlier_table.add_column("Eşik Yöntemi", style="bold cyan", width=18)
outlier_table.add_column("Euclidean", style="yellow", justify="center", width=12)
outlier_table.add_column("Mahalanobis", style="yellow", justify="center", width=12)

for method, counts in thresholds.items():
    euc_count = str(counts["euclidean"])
    mah_count = str(counts["mahalanobis"])
    outlier_table.add_row(method, euc_count, mah_count)

console.print(outlier_table)

# Örtüşme analizi - Top-10 karşılaştırması
print_section_header("TOP-10 ÖRTÜŞME ANALİZİ", "🎯")

top_10_euc_set = set(top_10_euc)
top_10_mah_set = set(top_10_mah)
overlap = top_10_euc_set.intersection(top_10_mah_set)
only_euc = top_10_euc_set - top_10_mah_set
only_mah = top_10_mah_set - top_10_euc_set

jaccard = len(overlap) / len(top_10_euc_set.union(top_10_mah_set))

overlap_table = Table(title="Top-10 Outlier Karşılaştırması", 
                      show_header=True, header_style="bold magenta",
                      border_style="green")
overlap_table.add_column("Kategori", style="bold cyan", width=30)
overlap_table.add_column("Sayı", style="yellow", justify="center", width=8)
overlap_table.add_column("Index'ler", style="white", width=30)

overlap_table.add_row("🔄 Ortak (Her ikisinde de)", str(len(overlap)), 
                      ", ".join(map(str, sorted(overlap))))
overlap_table.add_row("📐 Sadece Euclidean", str(len(only_euc)), 
                      ", ".join(map(str, sorted(only_euc))))
overlap_table.add_row("📏 Sadece Mahalanobis", str(len(only_mah)), 
                      ", ".join(map(str, sorted(only_mah))))
overlap_table.add_row("📊 Jaccard Benzerlik İndeksi", f"{jaccard:.2%}", "")

console.print(overlap_table)

# ═══════════════════════════════════════════════════════════════════════════
# KOVARYANSİN ETKİSİ ANALİZİ
# ═══════════════════════════════════════════════════════════════════════════

print_section_header("KOVARYANS ETKİSİ ANALİZİ", "🔬")

console.print(Panel(
    "[cyan]Mahalanobis mesafesinin farkı, kovaryans yapısını dikkate almasından gelir.\n"
    "Bu analiz, hangi özelliklerin outlier tespitinde daha etkili olduğunu gösterir.[/cyan]",
    border_style="blue"
))

# Eigenvalue analizi
eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
eigenvalues = eigenvalues[::-1]  # Büyükten küçüğe sırala
total_var = np.sum(eigenvalues)

eigen_table = Table(title="Kovaryans Matrisinin Özdeğer Analizi (PCA Perspektifi)",
                    show_header=True, header_style="bold magenta",
                    border_style="cyan")
eigen_table.add_column("Bileşen", style="bold cyan", justify="center", width=10)
eigen_table.add_column("Özdeğer", style="yellow", justify="right", width=12)
eigen_table.add_column("Varyans %", style="green", justify="right", width=12)
eigen_table.add_column("Kümülatif %", style="white", justify="right", width=12)

cumulative = 0
for i, ev in enumerate(eigenvalues, 1):
    var_pct = (ev / total_var) * 100
    cumulative += var_pct
    eigen_table.add_row(f"PC{i}", f"{ev:.4f}", f"{var_pct:.2f}%", f"{cumulative:.2f}%")

console.print(eigen_table)

# Condition number
condition_number = np.max(eigenvalues) / np.min(eigenvalues)
console.print(f"\n[bold]Kovaryans Matrisi Durum Sayısı:[/bold] {condition_number:.2f}")
if condition_number > 30:
    console.print("[red]⚠️ Yüksek multikolinearite tespit edildi![/red]")
else:
    console.print("[green]✓ Matris iyi durumda[/green]")

# ═══════════════════════════════════════════════════════════════════════════
# SINIF BAZLI ANALİZ
# ═══════════════════════════════════════════════════════════════════════════

print_section_header("SINIF BAZLI OUTLIER DAĞILIMI", "🏷️")

class_table = Table(title="Her Sınıf için Outlier Dağılımı (Top-10 içinde)",
                    show_header=True, header_style="bold magenta",
                    border_style="magenta")
class_table.add_column("Sınıf", style="bold cyan", width=12)
class_table.add_column("Euclidean", style="yellow", justify="center", width=12)
class_table.add_column("Mahalanobis", style="yellow", justify="center", width=12)
class_table.add_column("Toplam Örnek", style="dim", justify="center", width=12)

for i, sp in enumerate(species):
    euc_count = sum(1 for idx in top_10_euc if iris.target[idx] == i)
    mah_count = sum(1 for idx in top_10_mah if iris.target[idx] == i)
    total = sum(1 for t in iris.target if t == i)
    class_table.add_row(sp, str(euc_count), str(mah_count), str(total))

console.print(class_table)

# ═══════════════════════════════════════════════════════════════════════════
# KORELASYON ANALİZİ - İKİ MESAFE ARASINDAKİ İLİŞKİ
# ═══════════════════════════════════════════════════════════════════════════

print_section_header("MESAFELER ARASI KORELASYON", "📈")

pearson_corr, pearson_p = stats.pearsonr(euclidean_distances, mahal_distances)
spearman_corr, spearman_p = stats.spearmanr(euclidean_distances, mahal_distances)
kendall_corr, kendall_p = stats.kendalltau(euclidean_distances, mahal_distances)

corr_eval_table = Table(title="Euclidean ve Mahalanobis Mesafeleri Arasındaki Korelasyon",
                        show_header=True, header_style="bold magenta",
                        border_style="cyan")
corr_eval_table.add_column("Korelasyon Tipi", style="bold cyan", width=20)
corr_eval_table.add_column("Katsayı", style="green", justify="right", width=12)
corr_eval_table.add_column("p-değeri", style="yellow", justify="right", width=12)
corr_eval_table.add_column("Yorum", style="white", width=20)

def correlation_strength(r):
    r = abs(r)
    if r > 0.9: return "Çok Güçlü"
    if r > 0.7: return "Güçlü"
    if r > 0.5: return "Orta"
    if r > 0.3: return "Zayıf"
    return "Çok Zayıf"

corr_eval_table.add_row("Pearson", f"{pearson_corr:.4f}", f"{pearson_p:.2e}", correlation_strength(pearson_corr))
corr_eval_table.add_row("Spearman", f"{spearman_corr:.4f}", f"{spearman_p:.2e}", correlation_strength(spearman_corr))
corr_eval_table.add_row("Kendall Tau", f"{kendall_corr:.4f}", f"{kendall_p:.2e}", correlation_strength(kendall_corr))

console.print(corr_eval_table)

# ═══════════════════════════════════════════════════════════════════════════
# PERFORMANS DEĞERLENDİRMESİ VE ÖNERİLER
# ═══════════════════════════════════════════════════════════════════════════

print_section_header("SONUÇ VE ÖNERİLER", "✨")

evaluation_panel = f"""
[bold cyan]📊 GENEL DEĞERLENDİRME[/bold cyan]

[bold yellow]1. Mesafe Dağılımı:[/bold yellow]
   • Euclidean - Ortalama: {euclidean_distances.mean():.4f}, Std: {euclidean_distances.std():.4f}
   • Mahalanobis - Ortalama: {mahal_distances.mean():.4f}, Std: {mahal_distances.std():.4f}
   • Mahalanobis daha düşük varyans gösteriyor (daha stabil)

[bold yellow]2. Outlier Tespiti:[/bold yellow]
   • Top-10 örtüşme: {len(overlap)}/10 ({len(overlap)*10}%)
   • Jaccard benzerliği: {jaccard:.2%}
   • Farklı metrikler farklı outlier'ları tespit ediyor

[bold yellow]3. Kovaryans Etkisi:[/bold yellow]
   • Korelasyon matrisi yüksek bağımlılık gösteriyor
   • Mahalanobis bu yapıyı dikkate alarak daha doğru tespit yapıyor

[bold yellow]4. İstatistiksel Anlamlılık:[/bold yellow]
   • Chi² testi ile {np.sum(p_values < 0.05)} gerçek outlier tespit edildi (p<0.05)
   • Euclidean, yanlış pozitifler üretebilir

[bold green]✓ ÖNERİLER:[/bold green]

   1. [cyan]Multivariate verilerde Mahalanobis tercih edilmeli[/cyan]
   2. [cyan]Düşük korelasyonlu verilerde Euclidean yeterli[/cyan]
   3. [cyan]p-değeri bazlı eşikler daha güvenilir[/cyan]
   4. [cyan]Her iki metriğin birlikte kullanılması önerilir[/cyan]
"""

console.print(Panel(evaluation_panel, title="[bold]📋 Analiz Raporu[/bold]", border_style="green"))

# ═══════════════════════════════════════════════════════════════════════════
# GÖRSELLEŞTİRME
# ═══════════════════════════════════════════════════════════════════════════

print_section_header("GÖRSELLEŞTİRME OLUŞTURULUYOR", "🎨")

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Euclidean vs Mahalanobis Distance - Kapsamlı Analiz', fontsize=14, fontweight='bold')

# 1. Mesafe Dağılımı Histogramları
ax1 = axes[0, 0]
ax1.hist(euclidean_distances, bins=20, alpha=0.7, label='Euclidean', color='#3498db', edgecolor='white')
ax1.hist(mahal_distances, bins=20, alpha=0.7, label='Mahalanobis', color='#e74c3c', edgecolor='white')
ax1.axvline(euclidean_distances.mean(), color='#2980b9', linestyle='--', linewidth=2, label=f'Euc Mean: {euclidean_distances.mean():.2f}')
ax1.axvline(mahal_distances.mean(), color='#c0392b', linestyle='--', linewidth=2, label=f'Mah Mean: {mahal_distances.mean():.2f}')
ax1.set_xlabel('Mesafe')
ax1.set_ylabel('Frekans')
ax1.set_title('Mesafe Dağılımları')
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3)

# 2. Scatter Plot - İki mesafe karşılaştırması
ax2 = axes[0, 1]
colors = ['#2ecc71', '#3498db', '#9b59b6']
for i, sp in enumerate(species):
    mask = iris.target == i
    ax2.scatter(euclidean_distances[mask], mahal_distances[mask], 
                c=colors[i], label=sp, alpha=0.7, s=50, edgecolors='white', linewidth=0.5)

# Top-10 outlier'ları işaretle
for idx in top_10_euc:
    ax2.scatter(euclidean_distances[idx], mahal_distances[idx], 
                marker='o', s=150, facecolors='none', edgecolors='blue', linewidth=2)
for idx in top_10_mah:
    ax2.scatter(euclidean_distances[idx], mahal_distances[idx], 
                marker='s', s=150, facecolors='none', edgecolors='red', linewidth=2)

ax2.set_xlabel('Euclidean Mesafe')
ax2.set_ylabel('Mahalanobis Mesafe')
ax2.set_title('Mesafe Karşılaştırması')
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)

# Korelasyon çizgisi
z = np.polyfit(euclidean_distances, mahal_distances, 1)
p = np.poly1d(z)
ax2.plot(sorted(euclidean_distances), p(sorted(euclidean_distances)), 
         "k--", alpha=0.5, label=f'Korelasyon: {pearson_corr:.3f}')

# 3. Box Plot Karşılaştırması
ax3 = axes[0, 2]
box_data = [euclidean_distances, mahal_distances]
bp = ax3.boxplot(box_data, labels=['Euclidean', 'Mahalanobis'], patch_artist=True)
bp['boxes'][0].set_facecolor('#3498db')
bp['boxes'][1].set_facecolor('#e74c3c')
ax3.set_ylabel('Mesafe')
ax3.set_title('Box Plot Karşılaştırması')
ax3.grid(True, alpha=0.3)

# 4. Korelasyon Heatmap
ax4 = axes[1, 0]
sns.heatmap(corr_matrix, annot=True, cmap='RdYlBu_r', center=0, 
            xticklabels=['SL', 'SW', 'PL', 'PW'],
            yticklabels=['SL', 'SW', 'PL', 'PW'],
            ax=ax4, fmt='.2f', cbar_kws={'shrink': 0.8})
ax4.set_title('Özellik Korelasyon Matrisi')

# 5. Sınıf bazlı ortalama mesafeler
ax5 = axes[1, 1]
class_euc_means = [euclidean_distances[iris.target == i].mean() for i in range(3)]
class_mah_means = [mahal_distances[iris.target == i].mean() for i in range(3)]
x = np.arange(3)
width = 0.35
bars1 = ax5.bar(x - width/2, class_euc_means, width, label='Euclidean', color='#3498db', edgecolor='white')
bars2 = ax5.bar(x + width/2, class_mah_means, width, label='Mahalanobis', color='#e74c3c', edgecolor='white')
ax5.set_xlabel('Sınıf')
ax5.set_ylabel('Ortalama Mesafe')
ax5.set_title('Sınıf Bazlı Ortalama Mesafeler')
ax5.set_xticks(x)
ax5.set_xticklabels(species)
ax5.legend()
ax5.grid(True, alpha=0.3, axis='y')

# 6. Rank Karşılaştırması
ax6 = axes[1, 2]
euc_ranks = stats.rankdata(euclidean_distances)
mah_ranks = stats.rankdata(mahal_distances)
ax6.scatter(euc_ranks, mah_ranks, c=iris.target, cmap='viridis', alpha=0.6, s=30)
ax6.plot([0, 150], [0, 150], 'r--', alpha=0.5, label='Mükemmel Uyum')
ax6.set_xlabel('Euclidean Sıralaması')
ax6.set_ylabel('Mahalanobis Sıralaması')
ax6.set_title(f'Sıralama Karşılaştırması\n(Spearman ρ = {spearman_corr:.3f})')
ax6.legend()
ax6.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('euclidean_vs_mahalanobis_analysis.png', dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
plt.show()

console.print("[bold green]✓ Grafik 'euclidean_vs_mahalanobis_analysis.png' olarak kaydedildi![/bold green]")

# ═══════════════════════════════════════════════════════════════════════════
# FINAL
# ═══════════════════════════════════════════════════════════════════════════

console.print()
print_separator("═", 70)
console.print(Panel.fit(
    "[bold green]✓ KAPSAMLI ANALİZ TAMAMLANDI![/bold green]\n\n"
    "[cyan]Bu analiz, Euclidean ve Mahalanobis mesafelerinin outlier tespitindeki\n"
    "farklılıklarını ortaya koymakta ve her iki yöntemin güçlü/zayıf yönlerini\n"
    "karşılaştırmaktadır. Mahalanobis, yüksek korelasyonlu verilerde daha güvenilir\n"
    "sonuçlar verirken, Euclidean basitliği ve hızı ile avantaj sağlar.[/cyan]",
    border_style="green",
    padding=(1, 2)
))
print_separator("═", 70)