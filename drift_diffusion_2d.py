import numpy as np
import random
import matplotlib.pyplot as plt
import json


class GridModel2D_DD:
    """
    Class containing a rectangle room and can be used to simulate the drift-diffusion 2D equation given IC and BC
    """
    length = 0
    width = 0
    Nx = 0  # number of grid points in the x-direction
    Ny = 0  # number of grid points in the y-direction
    dx, dy = 0, 0  # distance between each grid point in x- or y-direction
    thermal_diffusivity = 19*10**(-6)  # constant value
    initial_temperature = 0  # temperature in the room
    temperature_outside = 0  # temperature outside
    heater_temperature = 0  # temperature of the heater
    heater_placement = 0  # the placement of the heater in the grid
    temperature_goal = 0  # the wanted average temperature in the room to end the simulation
    dt = 0  # the time step value
    temperature_matrix = []  # matrix containing the temperature at each grid point. [x,y] convection is used.
    temperature_matrix_previous_time = []  # help matrix to contain the previous temperature values
    time = 0  # value that keeps track of the time used in the simulation
    max_time = 10000000  # max limit of the time used to heat a room to temperature goal
    # velocity and change in velocity fields matrices
    v_x = []
    v_y = []
    a_x = []
    a_y = []

    def __init__(self, parameters):
        """
        Setting initial values for the simulation of the drift-diffusion 2d equation from json file
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
        self.dt = min(self.dx**2*self.dy**2/(2*self.thermal_diffusivity*(self.dx**2+self.dy**2)), 10)  # set dt to the minimum of 10 and max_dt to ensure stability
        self.temperature_matrix_previous_time = np.ones((self.Nx, self.Ny))*self.initial_temperature
        self.temperature_matrix_previous_time[self.heater_placement] = self.heater_temperature
        self.temperature_matrix = np.zeros_like(self.temperature_matrix_previous_time)

    def _temperature_at_new_timestep_ftcs_dd(self):
        """
        Function to compute the temperature at each grid point at a new time step using the FTCS scheme for drift-diffusion 2d equation
        """
        self.temperature_matrix[1:-1, 1:-1] = self.temperature_matrix_previous_time[1:-1, 1:-1] + self.thermal_diffusivity * self.dt * ((self.temperature_matrix_previous_time[2:, 1:-1] - 2 * self.temperature_matrix_previous_time[1:-1, 1:-1] + self.temperature_matrix_previous_time[:-2, 1:-1]) / self.dx ** 2 + (self.temperature_matrix_previous_time[1:-1, 2:] - 2 * self.temperature_matrix_previous_time[1:-1, 1:-1] + self.temperature_matrix_previous_time[1:-1, :-2]) / self.dy ** 2)
        self.temperature_matrix[1:-1, 1:-1] -= (self.temperature_matrix_previous_time[2:, 1:-1] - self.temperature_matrix_previous_time[:-2, 1:-1])*self.dt/(2*self.dx) * self.v_x[1:-1, 1:-1]
        self.temperature_matrix[1:-1, 1:-1] -= (self.temperature_matrix_previous_time[1:-1, 2:] - self.temperature_matrix_previous_time[1:-1, :-2])*self.dt/(2*self.dy) * self.v_y[1:-1, 1:-1]
        self.temperature_matrix[1:-1, 1:-1] -= self.dt*(self.temperature_matrix_previous_time[1:-1, 1:-1]*self.a_x[1:-1, 1:-1] + self.temperature_matrix_previous_time[1:-1, 1:-1]*self.a_y[1:-1, 1:-1])
        self.temperature_matrix[0, :] = (9 * self.temperature_matrix_previous_time[1, :] + self.temperature_outside) / 10  # wall update
        self.temperature_matrix[-1, :] = (9 * self.temperature_matrix_previous_time[-2, :] + self.temperature_outside) / 10  # wall update
        self.temperature_matrix[:, 0] = (9 * self.temperature_matrix_previous_time[:, 1] + self.temperature_outside) / 10  # wall update
        self.temperature_matrix[:, -1] = (9 * self.temperature_matrix_previous_time[:, -2] + self.temperature_outside) / 10  # wall update
        self.temperature_matrix[self.heater_placement] = self.heater_temperature  # reset the temperature of the heater
        self.temperature_matrix[self.temperature_matrix > self.heater_temperature] = self.heater_temperature  # keep max temperature at heater temperature
        self.temperature_matrix_previous_time = self.temperature_matrix
        self.time += self.dt  # increment in time for each time step

    def plot_temperature_room(self):
        """
        Function to plot the temperature in the room using the values contained in temperature_matrix
        """
        print("avg_temp in room: ", np.mean(self.temperature_matrix))
        print("Time: ", self.time)
        plt.imshow(self.temperature_matrix, cmap=plt.get_cmap('hot'), vmin=min(self.initial_temperature, self.temperature_outside), vmax=self.heater_temperature)
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

        self.time = 0  # reinitialize the time used in the simulation to be able to use object multiple times
        self.v_x, self.v_y = self.create_velocity_field()  # create velocity field
        self.v_x /= 1000  # scale values to avoid unphysical results
        self.v_y /= 1000  # scale values to avoid unphysical results

        self.a_x, self.a_y = self.create_change_in_velocity_field()  # create change in velocity field
        self.a_x /= 1000  # scale values to avoid unphysical results
        self.a_y /= 1000  # scale values to avoid unphysical results

        # reinitialization of the matrices containing the temperature to be able to use object multiple times
        self.temperature_matrix_previous_time = np.ones((self.Nx, self.Ny))*self.initial_temperature
        self.temperature_matrix_previous_time[self.heater_placement] = self.heater_temperature
        self.temperature_matrix = np.zeros_like(self.temperature_matrix_previous_time)
        # until the average temperature of the room is above the temperature goal take a new time step
        # also make use of max_time in the cases where the room never achieve the goal temperature
        while np.mean(self.temperature_matrix) < self.temperature_goal and self.time < self.max_time:
            self._temperature_at_new_timestep_ftcs_dd()
        print(self.time)
        return self.time

    def create_velocity_field(self):
        """
        Creates a velocity field to simulate a fan in all directions away from the heater.
        :return: matrices containing the velocity in all grid points in the x- and y-direction.
        """
        v_x = np.zeros((self.Nx, self.Ny))
        v_y = np.zeros_like(v_x)
        x_H, y_H = self.heater_placement[0], self.heater_placement[1]
        for i in range(self.Nx):
            for j in range(self.Ny):
                if (i, j) != (x_H, y_H):
                    v_x[i, j], v_y[i, j] = (i - x_H) / ((i - x_H) ** 2 + (j - y_H) ** 2), (j - y_H) / ((i - x_H) ** 2 + (j - y_H) ** 2)
        return v_x, v_y

    def create_change_in_velocity_field(self):
        """
        Creates a field containing the change in velocity for all directions away from the heater.
        :return: matrices containing the change in velocity in all grid points in the x- and y-direction.
        """
        a_x = np.zeros((self.Nx, self.Ny))
        a_y = np.zeros_like(a_x)
        x_H, y_H = self.heater_placement[0], self.heater_placement[1]
        for i in range(self.Nx):
            for j in range(self.Ny):
                if (i, j) != (x_H, y_H):
                    a_x[i, j], a_y[i, j] = ((j-y_H)**2-(i - x_H)**2) / ((i - x_H) ** 2 + (j - y_H) ** 2)**2, ((i-x_H)**2-(j - y_H)**2) / ((i - x_H) ** 2 + (j - y_H) ** 2)**2
        return a_x, a_y


# Example code which can be used to simulate the temperature in a room for a given json file and heater placement
# parametersAleks = json.load(open("initial_values/aleksander.json", "r"))
# heater_place = [12, 24]
# square_room = GridModel2D_DD(parametersAleks)
# square_room.simulate(heater_place)
# square_room.plot_temperature_room()
