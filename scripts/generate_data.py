"""Генерация синтетических тестовых данных (``data/*``).

Создаётся 2D задача Пуассона с пяти-точечной схемой на решётке m x p:

* ``A``      — матрица жёсткости (4 на диагонали, -1 у соседей по кресту);
* ``X_true`` — гладкое точное решение (произведение синусов);
* ``B_base`` — эталонная правая часть ``A @ X_true``;
* ``B``      — "измеренная" правая часть ``B_base + шум`` (малое отклонение,
  как в исходных данных задачи).

Каждая задача выводится во все поддерживаемые ``Files.read`` форматы:
``.lst``, ``.json``, ``.csv``, ``.npz`` (см. README).

Запуск::

    python scripts/generate_data.py
"""

import json
import os

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")


def poisson_matrix(m, p):
    """Матрица пяти-точечной схемы Лапласа на решётке m x p (узлов: m*p)."""
    n = m * p
    A = np.zeros((n, n))
    for i in range(m):
        for j in range(p):
            idx = i * p + j
            A[idx, idx] = 4.0
            if i > 0:
                A[idx, (i - 1) * p + j] = -1.0
            if i < m - 1:
                A[idx, (i + 1) * p + j] = -1.0
            if j > 0:
                A[idx, i * p + (j - 1)] = -1.0
            if j < p - 1:
                A[idx, i * p + (j + 1)] = -1.0
    return A


def exact_solution(m, p):
    """Гладкое точное решение — произведение синусов."""
    n = m * p
    x = np.zeros(n)
    for i in range(m):
        for j in range(p):
            x[i * p + j] = np.sin(np.pi * (i + 1) / (m + 1)) * np.sin(np.pi * (j + 1) / (p + 1))
    return x


def write_lst(path, name, A, B, B_base, X_true):
    """Записать задачу в формате ``.lst`` (нумерация с 1)."""
    n = A.shape[0]
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{name} {n}\n")
        for x_idx in range(n):
            nz = np.nonzero(A[x_idx])[0]
            f.write(f"{x_idx + 1} {B[x_idx]:.15g} {len(nz)}\n")
            for y_idx in nz:
                f.write(f"{y_idx + 1} {A[x_idx, y_idx]:.15g}\n")
        f.write(f"d {n}\n")
        for i in range(n):
            f.write(f"{i + 1} {B_base[i]:.15g} {X_true[i]:.15g}\n")


def write_json(path, A, B, B_base, X_true):
    """Записать задачу в формате ``.json``."""
    data = {
        "matrix": A.tolist(),
        "vector": B.tolist(),
        "result": X_true.tolist(),
        "vector_mod": B_base.tolist(),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


def write_csv(path, A, B, B_base, X_true):
    """Записать задачу в формате ``.csv`` (секции matrix/vector/result/vector_mod)."""
    with open(path, "w", encoding="utf-8") as f:
        f.write("matrix\n")
        for row in A:
            f.write(",".join(f"{v:.15g}" for v in row) + "\n")
        for section, values in (("vector", B), ("result", X_true), ("vector_mod", B_base)):
            f.write(f"{section}\n")
            f.write(",".join(f"{v:.15g}" for v in values) + "\n")


def write_npz(path, A, B, B_base, X_true):
    """Записать задачу в архив NumPy ``.npz`` (ключи A/b/x_ref/b_base)."""
    np.savez(path, A=A, b=B, x_ref=X_true, b_base=B_base)


def generate(name, m, p, seed=42, noise=1e-4):
    """Сгенерировать задачу ``data/<name>.{lst,json,csv,npz}`` на решётке m x p."""
    rng = np.random.RandomState(seed)
    A = poisson_matrix(m, p)
    X_true = exact_solution(m, p)
    B_base = A @ X_true
    B = B_base + noise * rng.randn(B_base.size)
    write_lst(os.path.join(DATA_DIR, f"{name}.lst"), name, A, B, B_base, X_true)
    write_json(os.path.join(DATA_DIR, f"{name}.json"), A, B, B_base, X_true)
    write_csv(os.path.join(DATA_DIR, f"{name}.csv"), A, B, B_base, X_true)
    write_npz(os.path.join(DATA_DIR, f"{name}.npz"), A, B, B_base, X_true)
    print(f"Создан {name}.{{lst,json,csv,npz}}: решётка {m}x{p}, узлов {A.shape[0]}")


if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)
    generate("frund_18_21", 18, 21)
    generate("frund_21_21", 21, 21)
    generate("frund", 10, 10)
