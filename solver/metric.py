"""Метрики качества решения: сравнение результата с эталонным вектором."""

import numpy as np


class Metric:
    """Критерии отличия двух векторов (результата и эталона).

    Поддерживаемые метрики:
    * ``discrepancy``    — разность ``base - result`` (остаток);
    * ``normal``         — евклидова норма вектора;
    * ``max_absolute``   — максимальное абсолютное отклонение;
    * ``max_relative``   — максимальное относительное отклонение;
    * ``min_relative``   — минимальное относительное отклонение;
    * ``avg_relative``   — среднее относительное отклонение.

    Все расчёты ведутся в ``float64``.
    """

    def __init__(self):
        self.disc = None      # разность base - result
        self.norm = 0.0       # норма
        self.max_abs = 0.0    # макс. абсолютное отклонение
        self.max_rel = 0.0    # макс. относительное отклонение
        self.min_rel = None   # мин. относительное отклонение
        self.avg_rel = 0.0    # среднее относительное отклонение

    def discrepancy(self, base, result):
        """Разность ``base - result`` (остаток)."""
        base = np.array(base, dtype=np.float64)
        result = np.array(result, dtype=np.float64)
        self.disc = base - result
        return self.disc

    def normal(self, vector):
        """Евклидова норма вектора."""
        self.norm = np.linalg.norm(vector)
        return self.norm

    def max_absolute(self, base, result):
        """Максимальное по модулю значение разности."""
        self.disc = self.discrepancy(base, result)
        n = len(self.disc)
        self.max_abs = 0
        for i in range(n):
            self.max_abs = max(abs(self.disc[i]), self.max_abs)
        return self.max_abs

    def max_relative(self, base, result):
        """Максимальное относительное отклонение |base - result| / |base|
        (с нулевыми компонентами base не считаем)."""
        self.disc = self.discrepancy(base, result)
        n = len(self.disc)
        self.max_rel = 0
        for i in range(n):
            if base[i] != 0:
                self.max_rel = max(abs(self.disc[i] / base[i]), self.max_rel)
        return self.max_rel

    def min_relative(self, base, result):
        """Минимальное относительное отклонение по тем же компонентам,
        что и в ``max_relative``."""
        self.disc = self.discrepancy(base, result)
        n = len(self.disc)
        self.min_rel = self.max_rel
        for i in range(n):
            if base[i] != 0:
                self.min_rel = min(abs(self.disc[i] / base[i]), self.min_rel)
        return self.min_rel

    def avg_relative(self, base, result):
        """Среднее относительное отклонение."""
        self.disc = self.discrepancy(base, result)
        n = len(self.disc)
        s = 0
        c = 0
        for i in range(n):
            if base[i] != 0:
                s += abs(self.disc[i] / base[i])
                c += 1
        if c > 0:
            self.avg_rel = s / c
        else:
            self.avg_rel = 0
        return self.avg_rel
