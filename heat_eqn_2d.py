import numpy as np
import random
import matplotlib.pyplot as plt
import json


class GridModel2D:
    """
    Class containing a rectangle room and can be used to simulate the 2D heat equation given an IC and BC
    """
    length = 0
    width = 0
    Nx = 0  # number of grid points in the x-direction
    Ny = 0  # number of grid points in the y-direction
    dx, dy = 0, 0  # spacing between grid points on the x- and y-axis
    thermal_diffusivity = 19*10**(-6)  # constant value
    initial_temperature = 0  # temperature of the room
    temperature_outside = 0  # temperature outside
    heater_temperature = 0  # temperature of the heater
    heater_placement = 0  # the placement of the heater in the grid
    temperature_goal = 0  # the wanted average temperature to end the simulation
    dt = 0  # time step value
    temperature_matrix = []
    temperature_matrix_previous_time = []
    time = 0  # value to keep track of time used in the simulation
    max_time = 10000000  # max limit of time allowed for the simulation

    def __init__(self, parameters):
        """
        Setting initial values for the simulation of the 2d heat equation from json file
        :param parameters: Dict from json file which includes values to be used in the simulation
        """
        self.length = parameters['simulation']['xlen']
        self.width = parameters['simulation']['ylen']
        self.Nx = parameters['simulation']['Nx']
        self.Ny = parameters['simulation']['Ny']
        self.initial_temperature = parameters['simulation']['initial_temperature']
        self.temperature_outside = parameters['simulation']['outside_temperature']
        self.heater_temperature = parameters['simulation']['heater_temperature']
        self.temperature_goal = parameters['simulation']['temperature_goal']
        self.dx, self.dy = self.length/(self.Nx-1), self.width/(self.Ny-1)
        self.dt = min(self.dx**2*self.dy**2/(2*self.thermal_diffusivity*(self.dx**2+self.dy**2)), 10)  # set dt to the minimum of 10 and max_dt to obtain stable solution
        self.temperature_matrix_previous_time = np.ones((self.Nx, self.Ny))*self.initial_temperature
        self.temperature_matrix_previous_time[self.heater_placement] = self.heater_temperature
        self.temperature_matrix = np.zeros_like(self.temperature_matrix_previous_time)

    def _temperature_at_new_timestep_ftcs(self):
        """
        Function to compute the temperature at each grid point at a new time step using the FTCS scheme for 2d heat equation
        """
        self.temperature_matrix[1:-1, 1:-1] = self.temperature_matrix_previous_time[1:-1, 1:-1] + self.thermal_diffusivity * self.dt * ((self.temperature_matrix_previous_time[2:, 1:-1] - 2 * self.temperature_matrix_previous_time[1:-1, 1:-1] + self.temperature_matrix_previous_time[:-2, 1:-1]) / self.dx ** 2 + (self.temperature_matrix_previous_time[1:-1, 2:] - 2 * self.temperature_matrix_previous_time[1:-1, 1:-1] + self.temperature_matrix_previous_time[1:-1, :-2]) / self.dy ** 2)
        self.temperature_matrix[0, :] = (9 * self.temperature_matrix_previous_time[1, :] + self.temperature_outside) / 10  # wall update
        self.temperature_matrix[-1, :] = (9 * self.temperature_matrix_previous_time[-2, :] + self.temperature_outside) / 10  # wall update
        self.temperature_matrix[:, 0] = (9 * self.temperature_matrix_previous_time[:, 1] + self.temperature_outside) / 10  # wall update
        self.temperature_matrix[:, -1] = (9 * self.temperature_matrix_previous_time[:, -2] + self.temperature_outside) / 10  # wall update
        self.temperature_matrix[self.heater_placement] = self.heater_temperature  # reset the temperature of the heater
        self.temperature_matrix_previous_time = self.temperature_matrix
        self.time += self.dt  # increment in time for each time step

    def plot_temperature_room(self):
        """
        Function to plot the temperature in the room using the values contained in temperature_matrix
        """
        print("avg_temp in room: ", np.mean(self.temperature_matrix))
        print("Time: ", self.time)
        plt.imshow(self.temperature_matrix, cmap=plt.get_cmap('hot'),
                   vmin=min(self.initial_temperature, self.temperature_outside), vmax=self.heater_temperature)
        plt.colorbar()
        plt.xlabel('y')
        plt.ylabel('x')
        plt.title('Simulated temperature from heater placement: {}'.format(self.heater_placement))
        plt.show()

    def simulate(self, heater_placement='Random'):
        """
        For a given heater_placement, uses FTCS to update the temperature at each time step until stopping criterion
        :param heater_placement: [x,y] array with the heater position, if not provided a random position is used
        :return: time used in the simulation
        """
        if heater_placement == 'Random':
            self.heater_placement = (random.randint(0, self.Nx-1), random.randint(0, self.Ny-1))
        else:
            self.heater_placement = (heater_placement[0], heater_placement[1])

        # reinitialize time and temperature matrices used in the simulation to be able to use the object multiple times
        self.time = 0
        self.temperature_matrix_previous_time = np.ones((self.Nx, self.Ny))*self.initial_temperature
        self.temperature_matrix_previous_time[self.heater_placement] = self.heater_temperature
        self.temperature_matrix = np.zeros_like(self.temperature_matrix_previous_time)
        # until the average temperature of the room is above the temperature goal take a new time step
        # also make use of max_time in the cases where the room never achieve the goal temperature
        while np.mean(self.temperature_matrix) < self.temperature_goal and self.time < self.max_time:
            self._temperature_at_new_timestep_ftcs()
        print(self.time)
        return self.time


# Example code which can be used to simulate the temperature in a room for a given json file and heater placement
# parameters = json.load(open("initial_values/quadratic.json", "r"))
# heater_place = [5, 5]
# square_room = GridModel2D(parameters)
# square_room.simulate(heater_place)
# square_room.plot_temperature_room()
