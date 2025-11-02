# ====================================================
#  OOP ile İstatistik + Optimizasyon + Reduce Örneği
# ====================================================

from functools import reduce
from scipy.stats import bernoulli, binom, poisson
from scipy.optimize import linprog

class ProbabilityCalculator:
    """Olasılık dağılımı hesaplamalarını yöneten sınıf."""
    def __init__(self, p, n, lam):
        self.p = p
        self.n = n
        self.lam = lam
        self.probabilities = {}

    def calculate_bernoulli(self, k=1):
        self.probabilities['Bernoulli_P1'] = bernoulli.pmf(k, self.p)
        return self.probabilities['Bernoulli_P1']

    def calculate_binomial(self, k=5):
        self.probabilities['Binom_P5'] = binom.pmf(k, self.n, self.p)
        return self.probabilities['Binom_P5']

    def calculate_poisson(self, k=2):
        self.probabilities['Poisson_P2'] = poisson.pmf(k, self.lam)
        return self.probabilities['Poisson_P2']

    def run_all_calculations(self):
        """Tüm olasılıkları hesaplar ve bir liste olarak döndürür."""
        return [
            self.calculate_bernoulli(),
            self.calculate_binomial(),
            self.calculate_poisson()
        ]

class OptimizationSolver:
    """Lineer programlama optimizasyonunu çözen sınıf."""
    def __init__(self, c, A_ub, b_ub, A_eq, b_eq, bounds):
        self.c = c
        self.A_ub = A_ub
        self.b_ub = b_ub
        self.A_eq = A_eq
        self.b_eq = b_eq
        self.bounds = bounds
        self.result = None

    def solve(self):
        """Optimizasyon problemini çözer ve sonucu saklar."""
        self.result = linprog(self.c, A_ub=self.A_ub, b_ub=self.b_ub,
                               A_eq=self.A_eq, b_eq=self.b_eq,
                               bounds=self.bounds, method='highs')
        return self.result

class AnalysisOrchestrator:
    """Analiz sürecini yönetir ve sonuçları özetler."""
    def __init__(self, prob_calc, opt_solver):
        self.prob_calc = prob_calc
        self.opt_solver = opt_solver
        self.summary = {}

    def run_analysis(self):
        """Tüm analiz adımlarını çalıştırır ve bir özet oluşturur."""
        # 1. Olasılıkları hesapla
        probabilities_list = self.prob_calc.run_all_calculations()
        print("--- 1. Olasılık Hesaplamaları ---")
        print(f"Hesaplanan olasılıklar: {probabilities_list}\n")

        # 2. Optimizasyonu çöz
        opt_result = self.opt_solver.solve()
        print("--- 2. Lineer Optimizasyon ---")
        if opt_result.success:
            print("Optimal çözüm bulundu:")
            print(f"x1 = {opt_result.x[0]:.2f}, x2 = {opt_result.x[1]:.2f}")
            print(f"Toplam maliyet: {opt_result.fun}\n")
        else:
            print(f"Optimizasyon başarısız: {opt_result.message}\n")

        # 3. Sonuçları birleştir ve özetle
        self.summary.update(self.prob_calc.probabilities)
        
        total_prob = reduce(lambda x, y: x + y, probabilities_list)
        self.summary["Average_Prob"] = total_prob / len(probabilities_list)
        
        if opt_result.success:
            self.summary["x1"] = opt_result.x[0]
            self.summary["x2"] = opt_result.x[1]
            self.summary["Total_Cost"] = opt_result.fun

    def display_summary(self):
        """Sonuç özetini ekrana yazdırır."""
        print("--- 3. Sonuç Özeti ---")
        for key, value in self.summary.items():
            print(f"{key}: {value}")

if __name__ == "__main__":
    # Parametreleri ve girdileri tanımla
    prob_calculator = ProbabilityCalculator(p=0.6, n=10, lam=3)
    
    opt_solver = OptimizationSolver(c=[10, 15], A_ub=[[1, 0], [0, 1]], b_ub=[8, 6],
                                    A_eq=[[1, 1]], b_eq=[10], bounds=[(0, None), (0, None)])

    # Analizi yönet ve çalıştır
    main_analysis = AnalysisOrchestrator(prob_calculator, opt_solver)
    main_analysis.run_analysis()
    main_analysis.display_summary()