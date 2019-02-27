import numpy as np
import matplotlib.pyplot as plt


class grid_modell_2d:
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
    dt = 0
    temperature_matrix = []
    temperature_matrix_previous_time = []

    def __init__(self, x_len, y_len, Nx, Ny, init_temp, heater_temp, outside_temp, heater_placement):
        self.length = x_len
        self.width = y_len
        self.Nx = Nx
        self.Ny = Ny
        self.initial_temperature = init_temp
        self.temperature_outside = outside_temp
        self.heater_temperature = heater_temp
        self.heater_placement = heater_placement
        self.dx, self.dy = x_len/(Nx-1), y_len/(Ny-1)
        self.dt = min(self.dx**2*self.dy**2/(2*self.thermal_diffusivity*(self.dx**2+self.dy**2)), 10)  # set dt to the minimum of 10 and max_dt to obtain stable solution
        self.temperature_matrix_previous_time = np.ones((self.Nx, self.Ny))*self.initial_temperature
        self.temperature_matrix_previous_time[self.heater_placement] = self.heater_temperature
        self.temperature_matrix = np.zeros_like(self.temperature_matrix_previous_time)

    def temperature_at_new_timestep_cds(self):
        # Propagate with forward-difference in time, central-difference in space
        self.temperature_matrix[1:-1, 1:-1] = self.temperature_matrix_previous_time[1:-1, 1:-1] + self.thermal_diffusivity * self.dt * ((self.temperature_matrix_previous_time[2:, 1:-1] - 2 * self.temperature_matrix_previous_time[1:-1, 1:-1] + self.temperature_matrix_previous_time[:-2, 1:-1]) / self.dx ** 2 + (self.temperature_matrix_previous_time[1:-1, 2:] - 2 * self.temperature_matrix_previous_time[1:-1, 1:-1] + self.temperature_matrix_previous_time[1:-1, :-2]) / self.dy ** 2)
        self.temperature_matrix[self.heater_placement] = self.heater_temperature
        self.temperature_matrix[0, :] = (9 * self.temperature_matrix_previous_time[1, :] + self.temperature_outside) / 10
        self.temperature_matrix[-1, :] = (9 * self.temperature_matrix_previous_time[-2, :] + self.temperature_outside) / 10
        self.temperature_matrix[:, 0] = (9 * self.temperature_matrix_previous_time[:, 1] + self.temperature_outside) / 10
        self.temperature_matrix[:, -1] = (9 * self.temperature_matrix_previous_time[:, -2] + self.temperature_outside) / 10
        self.temperature_matrix_previous_time = self.temperature_matrix

    def find_temperature_after_n_timesteps(self, n):
        for i in range(n):
            self.temperature_at_new_timestep_cds()
        Temp = self.temperature_matrix
        print("avg_temp: ", np.mean(Temp))
        plt.imshow(self.temperature_matrix, cmap=plt.get_cmap('hot'), vmin=self.initial_temperature, vmax=self.heater_temperature)
        plt.colorbar()
        plt.show()


temperature_outside = 20+273
initial_temperature, heater_temperature = 15+273, 30+273
x_len, y_len, Nx, Ny = 4, 4, 15, 15
placement = (5, 5)

square_room = grid_modell_2d(x_len, y_len, Nx, Ny, initial_temperature, heater_temperature, temperature_outside, placement)
#print(np.shape(square_room.temperature_matrix))
square_room.find_temperature_after_n_timesteps(10000000)
