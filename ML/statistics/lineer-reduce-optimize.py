# ====================================================
#  Temel İstatistik + Optimizasyon + Reduce Örneği
# ====================================================

from functools import reduce
from scipy.stats import bernoulli, binom, poisson
from scipy.optimize import linprog

# ----------------------------------------------------
# 1. Olasılık Dağılımları (Discrete Probability)
# ----------------------------------------------------
print("\n--- 1. Olasılık Dağılımları ---")

# Bernoulli: tek olay
p = 0.6
rv_bern = bernoulli(p)
print("Bernoulli P(X=1):", rv_bern.pmf(1))

# Binomial: 10 deneme, başarı olasılığı 0.6
n = 10
rv_binom = binom(n, p)
print("Binom P(X=5):", rv_binom.pmf(5))

# Poisson: Ortalama 3 olay
lam = 3
rv_pois = poisson(lam)
print("Poisson P(X=2):", rv_pois.pmf(2))

# Küçük bir olasılık listesi üretelim
probabilities = [rv_bern.pmf(1), rv_binom.pmf(5), rv_pois.pmf(2)]
print("Olasılıklar listesi:", probabilities)

# ----------------------------------------------------
# 2. Lineer Programlama (Linear Programming)
# ----------------------------------------------------
print("\n--- 2. Lineer Programlama ---")

# Basit üretim optimizasyonu: 2 ürün, kaynak sınırlı
# minimize 10*x1 + 15*x2
c = [10, 15]             # maliyetler
A_ub = [[1, 0], [0, 1]]  # her ürün max kapasite sınırı
b_ub = [8, 6]             # x1 <= 8, x2 <= 6
A_eq = [[1, 1]]           # toplam üretim talebi
b_eq = [10]               # x1 + x2 = 10
bounds = [(0, None), (0, None)]

res = linprog(c, A_ub=A_ub, b_ub=b_ub,
              A_eq=A_eq, b_eq=b_eq,
              bounds=bounds, method='highs')

if res.success:
    print("Optimal çözüm bulundu:")
    print(f"x1 = {res.x[0]:.2f}, x2 = {res.x[1]:.2f}")
    print("Toplam maliyet:", res.fun)
else:
    print("Optimizasyon başarısız:", res.message)

# ----------------------------------------------------
# 3. Reduce ile sonuçları özetlemek
# ----------------------------------------------------
print("\n--- 3. Reduce ile Özetleme ---")

# Olasılık listesinin ortalamasını hesapla
total_prob = reduce(lambda x, y: x + y, probabilities)
avg_prob = total_prob / len(probabilities)
print("Ortalama olasılık:", avg_prob)

# Optimizasyon değişkenlerini çarpım olarak birleştir
x_product = reduce(lambda x, y: x * y, res.x)
print("x1 * x2 =", x_product)

# Her şeyi tek çıktıda toparlayalım
summary = {
    "Bernoulli_P1": rv_bern.pmf(1),
    "Binom_P5": rv_binom.pmf(5),
    "Poisson_P2": rv_pois.pmf(2),
    "Average_Prob": avg_prob,
    "x1": res.x[0],
    "x2": res.x[1],
    "Total_Cost": res.fun
}
print("\nSonuç Özeti:", summary)