"""Организация сравнительного тестирования методов решения СЛАУ."""

import time

import numpy as np

from .files import Files
from .matrix import Matrix
from .metric import Metric


class Test:
    """Прогон всех методов на одной задаче.

    Считывает задачу (``Files.read``), для каждого метода (``Matrix``)
    решает СЛАУ, оценивает качество решения метриками (``Metric``),
    замеряет время на ``step`` решений и формирует сводную таблицу
    (``Files.write``).
    """

    def __init__(self):
        self.matrix = Matrix()
        self.files = Files()
        self.metric = Metric()

    def metric_stat(self, B, B_base):
        """Печать и возврат сводки метрик: норма, max abs, max/min/avg relative."""
        d = self.metric.discrepancy(B, B_base)
        norm = self.metric.normal(d)
        absolute = self.metric.max_absolute(B, B_base)
        rel_max = self.metric.max_relative(B, B_base)
        rel_min = self.metric.min_relative(B, B_base)
        rel_avg = self.metric.avg_relative(B, B_base)

        print("Отклонение:", d)
        print("Норма отклонения:", norm)
        print("Максимальное абсолютное отклонение:", absolute)
        print("Относительные отклонения (макс / мин / средн):", rel_max, rel_min, rel_avg)

        return norm, absolute, rel_max, rel_min, rel_avg

    def _evaluate_method(self, method_name, solver, A, B_base, mass_metric, repeat_count):
        """Оценка одного метода: решение, метрики качества и замер времени."""
        print(method_name)

        X = solver()
        B_new = A @ X
        B_new = np.array(B_new, dtype=np.float64)

        print("Норма решения: ", self.metric.normal(X))

        N, Abs, Rel_max, Rel_min, Rel_avg = self.metric_stat(B_new, B_base)

        mass_metric[0].append(method_name)
        mass_metric[1].append(N)
        mass_metric[2].append(Abs)
        mass_metric[3].append(Rel_max)
        mass_metric[4].append(Rel_min)
        mass_metric[5].append(Rel_avg)

        start = time.time()
        for _ in range(repeat_count):
            solver()
        end = time.time()
        elapsed = end - start
        print("Время", elapsed)
        mass_metric[6].append(elapsed)

    def run(self, name_r, name_w, step, regul_param=1e5):
        """Полный прогон всех методов на задаче ``name_r.lst``.

        * ``name_r``     — путь к задаче без расширения ``.lst``;
        * ``name_w``     — путь (без расширения) к файлу таблицы результатов;
        * ``step``       — число повторений для замера времени;
        * ``regul_param`` — параметр регуляризации.

        Таблица результатов (столбцы: метод, норма, max abs, max rel,
        min rel, avg rel, время) записывается в ``name_w.txt``.
        """
        print(name_r)

        # 0: метод, 1: норма, 2: abs_max, 3: rel_max, 4: rel_min, 5: rel_avg, 6: время
        mass_metric = [[], [], [], [], [], [], []]

        A, B, X, B_base = self.files.read(name_r)

        A = np.array(A, dtype=np.float64)
        B = np.array(B, dtype=np.float64)
        X = np.array(X, dtype=np.float64)
        B_base = np.array(B_base, dtype=np.float64)

        print(A)
        print(B)
        print(X)
        print(B_base)

        B_new = A @ X
        B_new = np.array(B_new, dtype=np.float64)
        print(B_new)

        print("Ранг матрицы: ", np.linalg.matrix_rank(A, 0.0000001))
        print("Норма эталонного решения: ", self.metric.normal(X))

        self.metric_stat(B, B_base)
        self.metric_stat(B_new, B_base)

        self._evaluate_method(
            "SVD",
            lambda: self.matrix.svd_res(A, B_base)[1],
            A, B_base, mass_metric, step
        )

        self._evaluate_method(
            "Regul",
            lambda: self.matrix.regul(A, B_base, regul_param),
            A, B_base, mass_metric, step
        )

        self._evaluate_method(
            "Pinv",
            lambda: self.matrix.pinv_res(A, B_base)[1],
            A, B_base, mass_metric, step
        )

        n = len(B_base)
        self._evaluate_method(
            "Gauss",
            lambda: self.matrix.GAUSSN(A, B_base, n),
            A, B_base, mass_metric, step
        )
        print(mass_metric)

        self.files.write(name_w, mass_metric,
                         ["Method", "Norma", "Absolute", "Relative", "Time"])
