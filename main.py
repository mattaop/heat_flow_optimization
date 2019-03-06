import BayesianOptimization as BO
import heat_eqn_2d as HE
import numpy as np
import json


print("Project in modelling and optimization of heat flow")


def load_initial_values(input_file):
    """
    Setting initial values for the optimization
    :param input_file: JSON file containing input data
    :return:
    """

    data = json.load(open(input_file, "r"))
    print(data)
    return data


def start_optimization():
    parameters = load_initial_values("initial_values/mathias.json")

    heater_placement = [0, 0]
    square_room = HE.GridModel2D(parameters)
    optimizing_algorithm = BO.BayesianOptimization(parameters)

    # Run the simulation for two random values to get samples for the optimization algorithm
    for i in range(3):
        time = square_room.simulate()
        print(optimizing_algorithm.xy_samples, optimizing_algorithm.t_samples)
        optimizing_algorithm.update_samples(square_room.heater_placement, time)

    # Run until we get convergence
    while not optimizing_algorithm.convergence():
        heater_placement = optimizing_algorithm.bayesian_optimization()
        time = square_room.simulate(heater_placement)
        optimizing_algorithm.update_samples(square_room.heater_placement, time)

    return heater_placement


if __name__ == '__main__':
    start_optimization()
