"""Точка входа: сравнение методов решения СЛАУ на тестовых данных.

Запуск::

    python main.py

Результаты записываются в каталог ``results/`` (по файлу ``<имя>_test.txt`` на задачу).
"""

import os

from solver import Test

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

# (имя задачи в data/, число повторений для замера времени)
TASKS = [
    ("frund_18_21", 1000),
    ("frund_21_21", 1000),
    ("frund", 1000),
]


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    t = Test()
    for name, repeats in TASKS:
        t.run(
            os.path.join(DATA_DIR, name),
            os.path.join(RESULTS_DIR, name + "_test"),
            repeats,
        )


if __name__ == "__main__":
    main()
