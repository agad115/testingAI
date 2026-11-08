"""Матричные методы решения систем линейных алгебраических уравнений (СЛАУ)."""

import numpy as np


class Matrix:
    """Набор методов решения СЛАУ A·x = b.

    Методы:
    * ``svd_res``    — решение через обратную матрицу, собранную по SVD;
    * ``regul``      — решение через симметричную регуляризацию (дополнение системы);
    * ``pinv_res``   — решение через псевдообратную матрицу Мора — Пенроуза;
    * ``GAUSSN``     — прямой метод Гаусса с выбором главного элемента;
    * ``invert_svd`` / ``invert_regul`` — восстановление обратной (псевдообратной)
      матрицы по чувствительности решения к возмущениям правой части.
    """

    def __init__(self):
        self.matrix1 = None  # матрица A
        self.matrix2 = None  # правая часть b
        self.step = None     # параметр регуляризации
        self.base = None     # базовое (опорное) решение
        self.koef = None      # величина возмущения правой части
        self.index = None     # индекс возмущаемой компоненты
        self.Dop1 = None      # дополненная матрица (regul)
        self.Dop2 = None      # дополненная правая часть (regul)
        self.A = None         # рабочая копия матрицы (GAUSSN)
        self.B = None         # рабочая правая часть (GAUSSN)
        self.X = None         # текущее решение

    # ------------------------------------------------------------------ SVD

    def svd_res(self, matrix1, matrix2):
        """Решить A·x = b, собрав обратную матрицу через SVD-разложение.

        A = U·S·Vt, обратная (псевдообратная) матрица: A^{-1} = Vt^T·S^{-1}·U^T,
        сингулярные значения <= 1e-4 считаются нулевыми (матрица вырожденная).
        """
        self.Dop1 = np.array(matrix1, dtype=np.float64)
        self.Dop2 = np.array(matrix2, dtype=np.float64)
        U, S, Vt = np.linalg.svd(self.Dop1)
        size = np.zeros(self.Dop1.shape)
        np.fill_diagonal(size, S)
        for i in range(min(size.shape)):
            if size[i][i] > 0.0001:
                size[i][i] = 1.0 / size[i][i]
        invert = np.transpose(Vt) @ size @ np.transpose(U)
        X = invert @ self.Dop2
        return invert, X

    # ------------------------------------------------------- регуляризация

    def regul(self, matrix1, matrix2, step):
        """Решить СЛАУ с симметричной регуляризацией.

        Система A (размер n+1) дополняется до квадратной (2n+1) системы:
        добавляются строки с единичной диагональю (матрица D, первая строка —1)
        и симметричные им столбцы; на диагонали правого нижнего блока
        выставляется параметр step. Возвращаются первые n+1 компонент решения.
        """
        n = len(matrix1) - 1
        self.Dop1 = np.array(matrix1, dtype=np.float64)
        self.Dop2 = np.array(matrix2, dtype=np.float64)
        D = np.array([[-1] * n], dtype=np.float64)
        k = np.identity(n, dtype=np.float64)
        D = np.concatenate((D, k), axis=0)
        Dt = np.transpose(D)
        Dt = np.concatenate((Dt, k), axis=1)
        self.Dop1 = np.concatenate((self.Dop1, D), axis=1)
        self.Dop1 = np.concatenate((self.Dop1, Dt), axis=0)
        Dop = np.array([0] * n, dtype=np.float64)
        self.Dop2 = np.concatenate((self.Dop2, Dop), axis=0)

        for j in range(n):
            self.Dop1[-j - 1][-j - 1] = step

        x = np.linalg.solve(self.Dop1, self.Dop2)

        return x[:n + 1]

    # -------------------------------------------------------- псевдообратная

    def pinv_res(self, matrix1, matrix2):
        """Решить A·x = b через псевдообратную матрицу Мора — Пенроуза (np.linalg.pinv)."""
        self.matrix1 = np.array(matrix1, dtype=np.float64)
        self.matrix2 = np.array(matrix2, dtype=np.float64)
        invert = np.linalg.pinv(self.matrix1)
        res = invert @ self.matrix2
        return invert, res

    # -------------------------------------- инверсия через возмущения правой части

    def calcul_param_svd(self, matrix1, matrix2, koef, index, base):
        """Столбец обратной матрицы через SVD-решение.

        Возмущаем index-ю компоненту правой части на ``koef`` и по приросту
        решения (X_new - base)/koef восстанавливаем соответствующий столбец A^{-1}.
        """
        self.base = base
        Dop = np.copy(self.matrix2)
        Dop[index] += koef
        X = self.svd_res(self.matrix1, Dop)[1]
        matrix3 = -(self.base - X) / koef
        return matrix3

    def invert_svd(self, matrix1, matrix2, koef):
        """Восстановить обратную (псевдообратную) матрицу методом SVD.

        По очереди возмущаем каждую компоненту правой части на ``koef``
        и собираем столбцы A^{-1}.
        """
        self.matrix1 = np.array(matrix1, dtype=np.float64)
        self.matrix2 = np.array(matrix2, dtype=np.float64)
        X1 = self.svd_res(self.matrix1, self.matrix2)[1]
        matrix3 = [self.calcul_param_svd(self.matrix1, self.matrix2, koef, 0, X1)]
        n = len(self.matrix2)
        for i in range(1, n):
            matrix3 = np.concatenate(
                (matrix3, [self.calcul_param_svd(self.matrix1, self.matrix2, koef, i, X1)]), axis=0)
        matrix3 = np.transpose(matrix3)
        return matrix3

    def calcul_param_regul(self, matrix1, matrix2, koef, index, base, step):
        """Столбец обратной матрицы через решение с регуляризацией.

        Аналог ``calcul_param_svd``, но опорное и возмущённое решения
        находятся методом ``regul`` с параметром ``step``.
        """
        self.base = base
        Dop = np.copy(self.matrix2)
        Dop[index] += koef
        X = self.regul(self.matrix1, Dop, step)
        matrix3 = -(self.base - X) / koef
        return matrix3

    def invert_regul(self, matrix1, matrix2, koef, step):
        """Восстановить обратную матрицу через метод регуляризации.

        По очереди возмущаем каждую компоненту правой части на ``koef``
        и собираем столбцы A^{-1} (схожа с ``invert_svd``).
        """
        self.matrix1 = np.array(matrix1, dtype=np.float64)
        self.matrix2 = np.array(matrix2, dtype=np.float64)
        X1 = self.regul(self.matrix1, self.matrix2, step)
        matrix3 = [self.calcul_param_regul(self.matrix1, self.matrix2, koef, 0, X1, step)]
        n = len(self.matrix2)
        for i in range(1, n):
            matrix3 = np.concatenate(
                (matrix3, [self.calcul_param_regul(self.matrix1, self.matrix2, koef, i, X1, step)]), axis=0)
        matrix3 = np.transpose(matrix3)
        return matrix3

    # --------------------------------------------------------- метод Гаусса

    def VEDELEMENT(self, a, b, n, k):
        """Выбор главного элемента в столбце ``k - 1``.

        Среди строк ``k..n-1`` ищем строку с максимальным по модулю элементом
        в столбце ``k-1`` и обмениваем её со строкой ``k-1`` (строки матрицы a
        и правая часть b меняются согласованно).
        """
        for i in range(k, n):
            if abs(a[i, k - 1]) > abs(a[k - 1, k - 1]):
                a[[k - 1, i]] = a[[i, k - 1]]
                b[k - 1], b[i] = b[i], b[k - 1]

    def GAUSSN(self, a, bb, n):
        """Решение системы уравнений методом Гаусса с выбором главного элемента.

        Векторизованная (NumPy) реализация; алгоритм и схема выбора
        главного элемента совпадают с исходной скалярной версией.
        Входная матрица не изменяется — работаем на копии.
        """
        self.A = np.array(a, dtype=np.float64)
        self.B = np.array(bb[:n], dtype=np.float64)
        self.X = np.zeros(n)

        # прямой ход
        for i in range(n - 1):
            self.VEDELEMENT(self.A, self.B, n, i + 1)
            pivot = self.A[i, i]
            if pivot == 0.0:
                pivot = 0.00001
                self.A[i, i] = pivot
            mult = self.A[i + 1:, i] / pivot
            sub = mult[:, None] * self.A[i, i:]
            self.A[i + 1:, i:] -= sub
            self.A[i + 1:, i] = 0.0
            self.B[i + 1:] -= mult * self.B[i]

        # обратный ход
        for i in range(n - 1, -1, -1):
            for j in range(n - 1, i, -1):
                self.B[i] -= self.A[i, j] * self.X[j]
            xx = self.A[i, i]
            if xx == 0.0:
                xx = 0.00000000007
            self.X[i] = self.B[i] / xx

        return self.X
