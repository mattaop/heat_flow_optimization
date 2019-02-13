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


def get_neighbours(position):
    grid_length = 4
    grid_width = 4
    neighbours = [np.add(position, (1, 0)), np.add(position, (0, 1)), np.subtract(position, (1, 0)), np.subtract(position, (0, 1))]
    neighbours = np.array([(i, j) for (i, j) in neighbours if i in range(0, grid_length - 1) and j in range(0, grid_width-1)])
    return neighbours


def gradient_descent():
    position = (0, 0)
    T = np.zeros((4, 4))
    while True:
        neighbours = get_neighbours(position)
        print(tuple(neighbours[0]))
        for i in range(len(neighbours)):
            T[tuple(neighbours[i])] = dummy_time(tuple(neighbours[i]))
        break

gradient_descent()
print(naive_search())
