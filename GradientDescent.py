import numpy as np
import heat_eqn_2d as HE
import drift_diffusion_2d as DD
import time


class GradientDescent:
    def __init__(self, parameters):
        self.grid_length = parameters['simulation']['Nx']
        self.grid_width = parameters['simulation']['Ny']
        self.parameters = parameters
        #self.room = HE.GridModel2D(parameters)
        self.room = DD.GridModel2D_DD(parameters)

    def get_neighbours(self, position):
        neighbours = [np.add(position, (1, 0)), np.add(position, (0, 1)), np.subtract(position, (1, 0)),
                      np.subtract(position, (0, 1))]
        neighbours = np.array(
            [(i, j) for (i, j) in neighbours if i in range(0, self.grid_length) and j in range(0, self.grid_width)])
        return neighbours

    def optimize(self):
        optimization_time = 0
        simulation_time = 0
        start_time = time.time()
        number_of_evals = 1
        positions = []
        times = []
        position = (np.random.randint(0, self.grid_length), np.random.randint(0, self.grid_width))
        #print('Random position generated to: ', position)
        T = np.zeros((self.grid_length, self.grid_width))
        T[position] = self.room.simulate(position)
        simulation_time += time.time()-start_time
        start_time = time.time()
        while True:
            neighbours = self.get_neighbours(position)
            #print('Neighbours are: ', neighbours)
            neighbours = map(tuple, neighbours)
            # print('Neighbours ', neighbours)
            improvement_found = False
            best_neighbour = position
            optimization_time += time.time() - start_time
            start_time = time.time()
            for neighbour in neighbours:
                #print('Currently evaluating neighbour: ', neighbour)
                T[neighbour] = self.room.simulate(neighbour)
                simulation_time += time.time() - start_time
                start_time = time.time()
                number_of_evals += 1
                # print('Neighbour is ', neighbour)
                #print('Time of neighbour is ', T[neighbour])
                if T[neighbour] < T[best_neighbour]:
                    improvement_found = True
                    best_neighbour = neighbour
            # print ('Time to heat cells: ', T)
            if not improvement_found:
                break
            position = best_neighbour
        positions.append(position)
        times.append(T[position])
        optimization_time += time.time() - start_time
        start_time = time.time()
        self.room.simulate(positions[np.argmin(times)])
        number_of_evals += 1
        simulation_time += time.time() - time.time()
        start_time = time.time()
        #self.room.plot_temperature_room()
        #return positions[np.argmin(times)], number_of_evals, optimization_time, simulation_time
        return positions[np.argmin(times)], number_of_evals, optimization_time, simulation_time
