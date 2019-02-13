# Her skjer det mykje kult
import numpy as np


def dummy_time(position):
    T = np.array([[43, 38, 38, 43], [37, 32, 32, 37], [37, 32, 32, 37], [43, 38, 38, 43]])
    return T[position]


def naive_search():
    T = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            T[i, j] = dummy_time((i, j))
    return T, np.unravel_index(np.argmin(T, axis=None), T.shape), np.min(T)


def get_neighbours():
    


def gradient_descent():
    position = (0, 0)
    while True:
        get_neighbours()

print(naive_search())
