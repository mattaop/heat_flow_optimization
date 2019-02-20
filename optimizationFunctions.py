# Her skjer det mykje kult
import numpy as np
np.random.seed(69)


def dummy_time(position):
    T = np.array([[43, 38, 38, 43], [37, 32, 32, 37], [2, 200, 0.2, 20], [43, 38, 38, 10]])
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
    neighbours = np.array([(i, j) for (i, j) in neighbours if i in range(0, grid_length) and j in range(0, grid_width)])
    return neighbours


def gradient_descent(iterations):
    positions = []
    times = []
    for iteration in range(iterations):
        position = (np.random.randint(0,4), np.random.randint(0,4))
        print('Random position generated to: ', position)
        T = np.zeros((4, 4))
        T[position] = dummy_time(position)
        while True:
            neighbours = get_neighbours(position)
            neighbours = map(tuple, neighbours)
            #print('Neighbours ', neighbours)
            improvement_found = False
            best_neighbour = position
            for neighbour in neighbours:
                T[neighbour] = dummy_time(neighbour)
                #print('Neighbour is ', neighbour)
                #print('Temp of neihgbour is ', T[neighbour])
                if T[neighbour] < T[best_neighbour]:
                    improvement_found = True
                    best_neighbour = neighbour
            #print ('Time to heat cells: ', T)
            if not improvement_found:
                break
            position = best_neighbour
        positions.append(position)
        times.append(T[position])
    return positions[np.argmin(times)]

print(gradient_descent(1))
#print('Entire grid: ', naive_search())
