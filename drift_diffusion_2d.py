import numpy as np
import random
import matplotlib.pyplot as plt
import json

class GridModel2D_DD:
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
    v_x = []
    v_y = []
    a_x = []
    a_y = []

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

    def _temperature_at_new_timestep_ftcs_dd(self):
        # Propagate with forward-difference in time, central-difference in space
        self.temperature_matrix[1:-1, 1:-1] = self.temperature_matrix_previous_time[1:-1, 1:-1] + self.thermal_diffusivity * self.dt * ((self.temperature_matrix_previous_time[2:, 1:-1] - 2 * self.temperature_matrix_previous_time[1:-1, 1:-1] + self.temperature_matrix_previous_time[:-2, 1:-1]) / self.dx ** 2 + (self.temperature_matrix_previous_time[1:-1, 2:] - 2 * self.temperature_matrix_previous_time[1:-1, 1:-1] + self.temperature_matrix_previous_time[1:-1, :-2]) / self.dy ** 2)
        self.temperature_matrix[1:-1, 1:-1] -= (self.temperature_matrix_previous_time[2:, 1:-1] - self.temperature_matrix_previous_time[:-2, 1:-1])*self.dt/(2*self.dx) * self.v_x[1:-1, 1:-1]
        self.temperature_matrix[1:-1, 1:-1] -= (self.temperature_matrix_previous_time[1:-1, 2:] - self.temperature_matrix_previous_time[1:-1, :-2])*self.dt/(2*self.dy) * self.v_y[1:-1, 1:-1]
        self.temperature_matrix[1:-1, 1:-1] -= self.dt*(self.temperature_matrix_previous_time[1:-1, 1:-1]*self.a_x[1:-1, 1:-1] + self.temperature_matrix_previous_time[1:-1, 1:-1]*self.a_y[1:-1, 1:-1])
        self.temperature_matrix[0, :] = (9 * self.temperature_matrix_previous_time[1, :] + self.temperature_outside) / 10
        self.temperature_matrix[-1, :] = (9 * self.temperature_matrix_previous_time[-2, :] + self.temperature_outside) / 10
        self.temperature_matrix[:, 0] = (9 * self.temperature_matrix_previous_time[:, 1] + self.temperature_outside) / 10
        self.temperature_matrix[:, -1] = (9 * self.temperature_matrix_previous_time[:, -2] + self.temperature_outside) / 10
        self.temperature_matrix[self.heater_placement] = self.heater_temperature
        self.temperature_matrix[self.temperature_matrix > self.heater_temperature] = self.heater_temperature
        self.temperature_matrix_previous_time = self.temperature_matrix
        self.time += self.dt

    def plot_temperature_room(self):
        print("avg_temp in room: ", np.mean(self.temperature_matrix))
        print("Time: ", self.time)
        plt.imshow(self.temperature_matrix, cmap=plt.get_cmap('hot'), vmin=min(self.initial_temperature, self.temperature_outside), vmax=self.heater_temperature)
        plt.colorbar()
        plt.show()

    def simulate(self, heater_placement='Random'):
        if heater_placement == 'Random':
            self.heater_placement = (random.randint(0, self.Nx-1), random.randint(0, self.Ny-1))
        else:
            self.heater_placement = (heater_placement[0], heater_placement[1])

        self.time = 0
        self.v_x, self.v_y = self.create_velocity_field()
        self.v_x /= 1000
        self.v_y /= 1000
        # self.v_x = np.ones((self.Nx, self.Ny))
        # self.v_y = np.ones((self.Nx, self.Ny))
        # self.v_x /= 1000
        # self.v_y /= 1000
        #print(np.max(self.v_x), np.min(self.v_x))
        self.a_x, self.a_y = self.create_change_in_velocity_field()
        self.a_x /= 1000
        self.a_y /= 1000
        #print(np.max(self.a_x), np.min(self.a_x))
        self.temperature_matrix_previous_time = np.ones((self.Nx, self.Ny))*self.initial_temperature
        self.temperature_matrix_previous_time[self.heater_placement] = self.heater_temperature
        self.temperature_matrix = np.zeros_like(self.temperature_matrix_previous_time)
        while np.mean(self.temperature_matrix) < self.temperature_goal and self.time < self.max_time:
            self._temperature_at_new_timestep_ftcs_dd()
        print(self.time)
        return self.time

    def create_velocity_field(self):
        v_x = np.zeros((self.Nx, self.Ny))
        v_y = np.zeros_like(v_x)
        x_H, y_H = self.heater_placement[0], self.heater_placement[1]
        #print(x_H, y_H)
        for i in range(self.Nx):
            for j in range(self.Ny):
                if (i, j) != (x_H, y_H):
                    v_x[i, j], v_y[i, j] = (i - x_H) / ((i - x_H) ** 2 + (j - y_H) ** 2), (j - y_H) / ((i - x_H) ** 2 + (j - y_H) ** 2)
        #print(v_x)
        #print(v_y)
        return v_x, v_y

    def create_change_in_velocity_field(self):
        a_x = np.zeros((self.Nx, self.Ny))
        a_y = np.zeros_like(a_x)
        x_H, y_H = self.heater_placement[0], self.heater_placement[1]
        #print(x_H, y_H)
        for i in range(self.Nx):
            for j in range(self.Ny):
                if (i, j) != (x_H, y_H):
                    a_x[i, j], a_y[i, j] = ((j-y_H)**2-(i - x_H)**2) / ((i - x_H) ** 2 + (j - y_H) ** 2)**2, ((i-x_H)**2-(j - y_H)**2) / ((i - x_H) ** 2 + (j - y_H) ** 2)**2
        #print(v_x)
        #print(v_y)
        return a_x, a_y


# Eksempel på bruk av kode
# parameters = json.load(open("initial_values/aleksander.json", "r"))
# heater_placement = [15, 15]
# square_room = GridModel2D_DD(parameters)
# square_room.simulate(heater_placement)
# print(square_room.temperature_matrix)
# square_room.plot_temperature_room()
