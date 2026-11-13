"""Точка входа: сравнение методов решения СЛАУ на тестовых данных.

Запуск::

    python main.py

Поддерживаются форматы задачи: ``.lst``, ``.json``, ``.csv``, ``.npz``
(расширение в имени задачи определяет формат).
Результаты записываются в каталог ``results/`` (по файлу ``<имя>_test.txt`` на задачу).
"""

import os

from solver import Test

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

# (путь к задаче, число повторений для замера времени)
# Число повторений уменьшено для больших решёток: метод Гаусса
# в векторизованном виде занимает ~0.6 с на решение при n=441.
TASKS = [
    (os.path.join(DATA_DIR, "frund.lst"), 1000),
    (os.path.join(DATA_DIR, "frund.json"), 1000),
    (os.path.join(DATA_DIR, "frund.csv"), 1000),
    (os.path.join(DATA_DIR, "frund.npz"), 1000),
    (os.path.join(DATA_DIR, "frund_18_21.lst"), 200),
    (os.path.join(DATA_DIR, "frund_21_21.lst"), 100),
]


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    t = Test()
    for path, repeats in TASKS:
        stem, ext = os.path.splitext(os.path.basename(path))
        out_name = f"{stem}_{ext[1:]}_test" if ext else f"{stem}_test"
        t.run(path, os.path.join(RESULTS_DIR, out_name), repeats)


if __name__ == "__main__":
    main()
