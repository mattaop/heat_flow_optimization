import numpy as np
from Simulation import drift_diffusion_2d as DD
from Simulation import heat_eqn_2d as HE
import time


class GradientDescent:
    """Class that contains methods for the Gradient Descent-based optimization."""
    def __init__(self, parameters, model):
        self.grid_length = parameters['simulation']['Nx']
        self.grid_width = parameters['simulation']['Ny']
        self.parameters = parameters

        if model == 'HE':
            self.room = HE.GridModel2D(parameters)
        if model == 'DD':
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
        times = [] # Store time used to heat the room
        positions = [] # Store position for heat source
        visited = np.zeros((self.grid_length, self.grid_width)) # Store visited positions to avoid simulating more than once for each
        for i in range(k):
            optimization_time = 0
            simulation_time = 0
            start_time = time.time()
            number_of_evals = 1
            position = (np.random.randint(0, self.grid_length), np.random.randint(0, self.grid_width))
            print('Start position generated to: ', position)
            T = np.zeros((self.grid_length, self.grid_width))
            T[position] = self.room.simulate(heater_placement=position, velocity_field='directional')
            visited[position[0]][position[1]] = 1
            simulation_time += time.time()-start_time
            start_time = time.time()

            # Run until convergence criterion is met (no neighbors yield better objective value)
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
                    if visited[neighbour[0]][neighbour[1]] == 1:
                        continue
                    T[neighbour] = self.room.simulate(heater_placement=neighbour, velocity_field='directional')
                    visited[neighbour[0]][neighbour[1]] = 1
                    simulation_time += time.time() - start_time
                    start_time = time.time()
                    number_of_evals += 1
                    print('Time to heat with position', neighbour, 'is', T[neighbour])
                    if T[neighbour] < T[best_neighbour]:
                        improvement_found = True
                        best_neighbour = neighbour
                # If none of the adjacent cells yield a better objective value, then stop
                if not improvement_found:
                    break
                position = best_neighbour

            positions.append(position)
            times.append(T[position])
            optimization_time += time.time() - start_time
            start_time = time.time()
            number_of_evals += 1
            simulation_time += time.time() - start_time
            #self.room.plot_temperature_room()
        return positions[np.argmin(times)], number_of_evals, optimization_time, simulation_time
