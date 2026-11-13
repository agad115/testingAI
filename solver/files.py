"""Чтение и запись файлов данных и результатов в нескольких форматах.

Поддерживаемые форматы задачи (расширение определяет формат):

* ``.lst``  — исходный текстовый формат (разреженная строка по узлам);
* ``.json`` — единый JSON-объект с ключами ``matrix``, ``vector``,
  ``result``, ``vector_mod``;
* ``.csv``  — секции ``matrix`` / ``vector`` / ``result`` / ``vector_mod``,
  значения разделены запятой;
* ``.npz``  — архив NumPy с массивами ``A``, ``b``, ``x_ref``, ``b_base``.

Форматы результата (таблица метрик): ``.txt`` (по умолчанию), ``.csv``,
``.json``.
"""

import csv
import json
import os

import numpy as np

#: ключи результата чтения: (матрица A, правая часть B, эталонное решение X,
#: эталонная правая часть B_base)
_READERS = (".lst", ".json", ".csv", ".npz")


class Files:
    """Работа с файлами задачи и таблицами результатов.

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

    # ------------------------------------------------------------------ чтение

    def read(self, name):
        """Прочитать файл задачи; формат определяется расширением.

        В ``name`` можно передать путь без расширения — тогда ожидается ``.lst``.
        Возвращает кортеж ``(matrix, vector, result, vector_mod)``:
        матрица A, правая часть B, эталонное решение X,
        эталонная правая часть B_base.
        """
        path = name if os.path.splitext(name)[1] else f"{name}.lst"
        ext = os.path.splitext(path)[1].lower()
        if ext not in _READERS:
            raise ValueError(
                f"Неподдерживаемый формат файла {path!r}; "
                f"поддерживаются: {', '.join(_READERS)}")
        reader = {
            ".lst": self._read_lst,
            ".json": self._read_json,
            ".csv": self._read_csv,
            ".npz": self._read_npz,
        }[ext]
        self.matrix, self.vector, self.result, self.vector_mod = reader(path)
        return self.matrix, self.vector, self.result, self.vector_mod

    def _read_lst(self, path):
        """Чтение исходного формата ``.lst`` (нумерация с 1)."""
        matrix, vector, result, vector_mod = [], [], [], []
        with open(path) as f:
            n = int(f.readline().split()[-1])
            matrix = [[0 for _ in range(n)] for _ in range(n)]
            for _ in range(n):
                par = f.readline().split()[-3:]
                x = int(par[0]) - 1
                vector.append(float(par[1]))
                k = int(par[2])
                for _ in range(k):
                    coord = f.readline().split()
                    y = int(coord[0]) - 1
                    matrix[x][y] = float(coord[1])

            d = int(f.readline().split()[1])
            for _ in range(d):
                s = f.readline().split()
                vector_mod.append(float(s[1]))
                result.append(float(s[2]))
        return matrix, vector, result, vector_mod

    def _read_json(self, path):
        """Чтение JSON: ``{"matrix": ..., "vector": ..., "result": ..., "vector_mod": ...}``."""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        matrix = data["matrix"]
        vector = data["vector"]
        result = data.get("result", [0.0] * len(vector))
        vector_mod = data.get("vector_mod", [0.0] * len(vector))
        return matrix, vector, list(result), list(vector_mod)

    def _read_csv(self, path):
        """Чтение CSV с секциями ``matrix`` / ``vector`` / ``result`` / ``vector_mod``."""
        matrix, vector, result, vector_mod = [], [], [], []
        section = None
        with open(path, encoding="utf-8") as f:
            for row in csv.reader(f):
                if not row:
                    continue
                key = row[0].strip().lower()
                if key in ("matrix", "vector", "result", "vector_mod"):
                    section = key
                    continue
                if section == "matrix":
                    matrix.append([float(v) for v in row])
                elif section == "vector":
                    vector.extend(float(v) for v in row)
                elif section == "result":
                    result.extend(float(v) for v in row)
                elif section == "vector_mod":
                    vector_mod.extend(float(v) for v in row)
        if section is None:
            raise ValueError(f"В файле {path!r} нет секций (matrix/vector/result/vector_mod)")
        return matrix, vector, result, vector_mod

    def _read_npz(self, path):
        """Чтение архива NumPy: ключи ``A``, ``b``, ``x_ref``, ``b_base``."""
        with np.load(path) as npz:
            matrix = npz["A"].tolist()
            vector = npz["b"].tolist()
            result = npz["x_ref"].tolist()
            vector_mod = npz["b_base"].tolist()
        return matrix, vector, result, vector_mod

    # ------------------------------------------------------------------ запись

    def write(self, name, mass_data, title, fmt=None):
        """Записать таблицу результатов в файл ``name``.

        ``mass_data`` — список списков (столбцов); каждая строка таблицы
        получаетcя из i-х элементов всех списков.

        Формат вывода по ``fmt`` или расширению ``name``
        (``.txt`` — столбцы через табуляцию, по умолчанию; ``.csv``; ``.json``).
        Если у ``name`` нет нужного расширения, оно дописывается.
        """
        self.name = name
        self.result = mass_data
        self.vector = title
        if fmt is None:
            fmt = os.path.splitext(name)[1].lower() or ".txt"
        elif not fmt.startswith("."):
            fmt = "." + fmt
        if not name.lower().endswith(fmt):
            name = name + fmt
        writer = {
            ".txt": self._write_txt,
            ".csv": self._write_csv,
            ".json": self._write_json,
        }[fmt]
        writer(name, mass_data, title)

    def _write_txt(self, path, mass_data, title):
        with open(path, "w") as f:
            f.write("\t".join(title) + "\n")
            for i in range(len(mass_data[0])):
                f.write("\t".join(str(mass_data[j][i]) for j in range(len(mass_data))) + "\n")

    def _write_csv(self, path, mass_data, title):
        with open(path, "w", newline="") as f:
            f.write(",".join(title) + "\n")
            for i in range(len(mass_data[0])):
                f.write(",".join(str(mass_data[j][i]) for j in range(len(mass_data))) + "\n")

    def _write_json(self, path, mass_data, title):
        data = {
            "columns": title,
            "rows": [[mass_data[j][i] for j in range(len(mass_data))]
                     for i in range(len(mass_data[0]))],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
