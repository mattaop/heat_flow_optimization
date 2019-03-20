import numpy as np
import random
import matplotlib.pyplot as plt


class GridModel2D:
    length = 0
    width = 0
    Nx = 0
    Ny = 0
    dx, dy = 0, 0
    thermal_diffusivity = 19*10**(-6)
    initial_temperature = 0
    temperature_outside = 0
    heater_temperature = 0
    heater_placement = 0
    temperature_goal = 0
    dt = 0
    temperature_matrix = []
    temperature_matrix_previous_time = []
    time = 0
    max_time = 10000000

    def __init__(self, parameters):
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
        # Propagate with forward-difference in time, central-difference in space
        self.temperature_matrix[1:-1, 1:-1] = self.temperature_matrix_previous_time[1:-1, 1:-1] + self.thermal_diffusivity * self.dt * ((self.temperature_matrix_previous_time[2:, 1:-1] - 2 * self.temperature_matrix_previous_time[1:-1, 1:-1] + self.temperature_matrix_previous_time[:-2, 1:-1]) / self.dx ** 2 + (self.temperature_matrix_previous_time[1:-1, 2:] - 2 * self.temperature_matrix_previous_time[1:-1, 1:-1] + self.temperature_matrix_previous_time[1:-1, :-2]) / self.dy ** 2)
        self.temperature_matrix[0, :] = (9 * self.temperature_matrix_previous_time[1, :] + self.temperature_outside) / 10
        self.temperature_matrix[-1, :] = (9 * self.temperature_matrix_previous_time[-2, :] + self.temperature_outside) / 10
        self.temperature_matrix[:, 0] = (9 * self.temperature_matrix_previous_time[:, 1] + self.temperature_outside) / 10
        self.temperature_matrix[:, -1] = (9 * self.temperature_matrix_previous_time[:, -2] + self.temperature_outside) / 10
        self.temperature_matrix[self.heater_placement] = self.heater_temperature
        self.temperature_matrix_previous_time = self.temperature_matrix
        self.time += self.dt

    def plot_temperature_room(self):
        print("avg_temp in room: ", np.mean(self.temperature_matrix))
        print("Time: ", self.time)
        plt.imshow(self.temperature_matrix, cmap=plt.get_cmap('hot'), vmin=self.initial_temperature, vmax=self.heater_temperature)
        plt.colorbar()
        plt.show()

    def simulate(self, heater_placement='Random'):
        if heater_placement == 'Random':
            self.heater_placement = (random.randint(0, self.Nx-1), random.randint(0, self.Ny-1))
        else:
            self.heater_placement = (heater_placement[0], heater_placement[1])

        self.time = 0
        self.temperature_matrix_previous_time = np.ones((self.Nx, self.Ny))*self.initial_temperature
        self.temperature_matrix_previous_time[self.heater_placement] = self.heater_temperature
        self.temperature_matrix = np.zeros_like(self.temperature_matrix_previous_time)
        while np.mean(self.temperature_matrix) < self.temperature_goal and self.time < self.max_time:
            self._temperature_at_new_timestep_ftcs()
        print(np.mean(self.temperature_matrix), self.time, self.heater_placement)
        return self.time


# Eksempel på bruk av kode
# temperature_outside = 20+273
# initial_temperature, heater_temperature = 15+273, 30+273
# x_len, y_len, Nx, Ny = 5, 5, 10, 10
# placement = (5, 5)
#
# number_timesteps = 1000
#
# square_room = GridModel2D(x_len, y_len, Nx, Ny, initial_temperature, heater_temperature, temperature_outside, placement)
# square_room.find_temperature_after_n_timesteps(number_timesteps)
#
# square_room.heater_placement = (1, 1)
# square_room.time = 0
# square_room.temperature_matrix_previous_time = np.ones((square_room.Nx, square_room.Ny))*square_room.initial_temperature
# square_room.temperature_matrix_previous_time[square_room.heater_placement] = square_room.heater_temperature
# square_room.find_temperature_after_n_timesteps(number_timesteps-10)

