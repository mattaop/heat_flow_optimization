import BayesianOptimization as BO
import heat_eqn_2d as HE
import numpy as np

print("Project in modelling and optimization of heat flow")

def initial_values():
    """
    Setting initial values for the optimization
    :return:
    """
    temperature_outside = 20 + 273
    initial_temperature, heater_temperature = 15 + 273, 30 + 273
    x_len, y_len, Nx, Ny = 4, 4, 15, 15
    return temperature_outside, initial_temperature, heater_temperature, x_len, y_len, Nx, Ny


def start_optimization():
    temperature_outside, initial_temperature, heater_temperature, x_len, y_len, Nx, Ny = initial_values()
    optimizing = True

    while optimizing:



if __name__ == '__main__':
    start_optimization()