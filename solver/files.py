"""Чтение и запись файлов данных и результатов."""


class Files:
    """Работа с файлами задачи (``*.lst``) и таблицами результатов (``*.txt``).

    Формат ``*.lst``::

        <имя> N                    # 1-я строка: последним числом — число узлов N
        <x+1> <B[x]> <k>           # узел: номер, значение правой части, число ненулевых
                                   #   элементов строки матрицы
        <y+1> <A[x][y]>            # k строк: столбец и значение ненулевого элемента
        d D                        # число D строк эталонных данных
        <i+1> <B_base[i]> <X[i]>   # эталонная правая часть и эталонное решение

    Числа в файле нумеруются с 1 (как в исходных данных задачи).
    """

    def __init__(self):
        self.name = None
        self.types = None
        self.matrix = []
        self.vector = []
        self.result = []
        self.vector_mod = []

    def read(self, name):
        """Прочитать файл ``{name}.lst``.

        Возвращает:
        * ``matrix``     — матрица A (список списков);
        * ``vector``     — правая часть B ("измеренная");
        * ``result``     — эталонное решение X;
        * ``vector_mod`` — эталонная правая часть B_base.
        """
        self.matrix = []
        self.vector = []
        self.result = []
        self.vector_mod = []
        with open(f"{name}.lst") as f:
            n = int(f.readline().split()[-1])
            for _ in range(n):
                self.matrix.append([0 for _ in range(n)])
            for _ in range(n):
                par = f.readline().split()[-3:]
                x = int(par[0]) - 1
                self.vector.append(float(par[1]))
                k = int(par[2])
                for _ in range(k):
                    coord = f.readline().split()
                    y = int(coord[0]) - 1
                    self.matrix[x][y] = float(coord[1])

            d = int(f.readline().split()[1])
            for _ in range(d):
                s = f.readline().split()
                self.vector_mod.append(float(s[1]))
                self.result.append(float(s[2]))
        return self.matrix, self.vector, self.result, self.vector_mod

    def write(self, name, mass_data, title):
        """Записать таблицу результатов в ``{name}.txt``.

        ``mass_data`` — список списков (столбцов); каждая строка таблицы
        получается из i-х элементов всех списков.
        """
        self.name = name
        self.result = mass_data
        self.vector = title
        with open(f"{self.name}.txt", "w") as f:
            f.write("\t".join(self.vector) + "\n")
            for i in range(len(self.result[0])):
                mass_str = []
                for j in range(len(self.result)):
                    mass_str.append(self.result[j][i])
                f.write("\t".join(list(map(str, mass_str))) + "\n")
