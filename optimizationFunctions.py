# Her skjer det mykje kult
import numpy as np
import heat_eqn_2d as HE
import json


np.random.seed(69)

def load_initial_values(input_file):
    """
    Setting initial values for the optimization
    :param input_file: Path to json file
    :return: Dict containing data from input_file
    """

    data = json.load(open(input_file, "r"))
    return data

def dummy_time(position):
    T = np.array([[43, 38, 38, 43], [37, 32, 32, 37], [2, 200, 0.2, 20], [43, 38, 38, 10]])
    return T[position]


def naive_search():
    parameters = load_initial_values("initial_values/mathias.json")
    square_room = HE.GridModel2D(parameters)
    T = np.zeros([parameters['simulation']['Nx'], parameters['simulation']['Ny']])
    for i in range(parameters['simulation']['Nx']):
        for j in range(parameters['simulation']['Ny']):
            T[i, j] = square_room.simulate([i, j])
            print(i, j)
    return T, np.unravel_index(np.argmin(T, axis=None), T.shape), np.min(T)


def get_neighbours(position):
    grid_length = 4
    grid_width = 4
    neighbours = [np.add(position, (1, 0)), np.add(position, (0, 1)), np.subtract(position, (1, 0)), np.subtract(position, (0, 1))]
    neighbours = np.array([(i, j) for (i, j) in neighbours if i in range(0, grid_length) and j in range(0, grid_width)])
    return neighbours


def gradient_descent(iterations):
    parameters = load_initial_values("initial_values/mathias.json")
    positions = []
    times = []
    square_room = HE.GridModel2D(parameters)
    for iteration in range(iterations):
        position = (np.random.randint(0, parameters['simulation']['Nx']), np.random.randint(0, parameters['simulation']['Ny']))
        print('Random position generated to: ', position)
        T = np.zeros((parameters['simulation']['Nx'], parameters['simulation']['Ny']))

        T[position] = square_room.simulate(position)
        while True:
            neighbours = get_neighbours(position)
            neighbours = map(tuple, neighbours)
            print('Neighbours ', neighbours)
            improvement_found = False
            best_neighbour = position
            for neighbour in neighbours:
                T[neighbour] = square_room.simulate(neighbour)
                print('Neighbour is ', neighbour)
                print('Time of neihgbour is ', T[neighbour])
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




T, min_T_arg, min_T = naive_search()
print(T)
#print(gradient_descent(10))
#print('Entire grid: ', naive_search())

