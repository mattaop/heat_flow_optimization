import numpy as np


class GradientDescent:
    def __init__(self, parameters):
        pass

    def _get_neighbours(self, position):
        grid_length = 4
        grid_width = 4
        neighbours = [np.add(position, (1, 0)), np.add(position, (0, 1)), np.subtract(position, (1, 0)), np.subtract(position, (0, 1))]
        neighbours = np.array([(i, j) for (i, j) in neighbours if i in range(0, grid_length) and j in range(0, grid_width)])
        return neighbours

    def optimize(self, ):
        positions = []
        times = []
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
