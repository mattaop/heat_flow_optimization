import numpy as np
import heat_eqn_2d as HE
import drift_diffusion_2d as DD
import time


class GradientDescent:
    """Class that contains methods for the Gradient Descent-based optimization."""
    def __init__(self, parameters):
        self.grid_length = parameters['simulation']['Nx']
        self.grid_width = parameters['simulation']['Ny']
        self.parameters = parameters
        #self.room = HE.GridModel2D(parameters)
        self.room = DD.GridModel2D_DD(parameters)

    def get_neighbours(self, position):
        """
        Generate non-diagonally adjacent cells.
        :param position: Coordinates of the position as (x,y).
        :return: Array containing all non-diagonally adjacent cells.
        """
        neighbours = [np.add(position, (1, 0)), np.add(position, (0, 1)), np.subtract(position, (1, 0)),
                      np.subtract(position, (0, 1))]
        neighbours = np.array(
            [(i, j) for (i, j) in neighbours if i in range(0, self.grid_length) and j in range(0, self.grid_width)])
        return neighbours

    def optimize(self, k):
        """
        Start at a random position and move in the direction with largest improvement until we reach a optimum. Run k times, and return the best result.
        :param k: integer number of times the optimization method should run.
        :return: optimal position, number of times heat simulation was run, computation time spent on optimization and simulation.
        """
        times = []
        positions = []
        for i in range(k):
            optimization_time = 0
            simulation_time = 0
            start_time = time.time()
            number_of_evals = 1
            position = (np.random.randint(0, self.grid_length), np.random.randint(0, self.grid_width))
            #print('Random start position generated to: ', position)
            T = np.zeros((self.grid_length, self.grid_width))
            T[position] = self.room.simulate(heater_placement = position, velocity_field = 'directional')
            simulation_time += time.time()-start_time
            start_time = time.time()
            while True:
                neighbours = self.get_neighbours(position)
                print('Current position:', position)
                print('Neighbours are:', neighbours)
                neighbours = map(tuple, neighbours)
                improvement_found = False
                best_neighbour = position
                optimization_time += time.time() - start_time
                start_time = time.time()
                for neighbour in neighbours:
                    #print('Currently evaluating neighbour:', neighbour)
                    T[neighbour] = self.room.simulate(heater_placement = neighbour, velocity_field = 'directional')
                    simulation_time += time.time() - start_time
                    start_time = time.time()
                    number_of_evals += 1
                    print('Time to heat for neighbour is:', T[neighbour])
                    if T[neighbour] < T[best_neighbour]:
                        improvement_found = True
                        best_neighbour = neighbour
                # If none of the adjacent cells yield a better objective value, stop
                if not improvement_found:
                    break
                position = best_neighbour
            positions.append(position)
            times.append(T[position])
            optimization_time += time.time() - start_time
            start_time = time.time()
            #self.room.simulate(positions[np.argmin(times)])
            number_of_evals += 1
            simulation_time += time.time() - start_time()
            #self.room.plot_temperature_room()
        return positions[np.argmin(times)], number_of_evals, optimization_time, simulation_time
