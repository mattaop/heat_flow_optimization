import BayesianOptimization as BO
import heat_eqn_2d as HE
import test
import numpy as np
import random
import json


print("Project in modelling and optimization of heat flow")


def load_initial_values(input_file):
    """
    Setting initial values for the optimization
    :param input_file: Path to json file
    :return: Dict containing data from input_file
    """

    data = json.load(open(input_file, "r"))
    return data


def scale_time(time):
    return 1-time/10000000


def start_optimization():
    parameters = load_initial_values("initial_values/mathias.json")

    heater_placement = [0, 0]
    square_room = HE.GridModel2D(parameters)
    optimizing_algorithm = BO.BayesianOptimization(parameters)
    test_function = test.TestFunction()

    # Run the simulation for two random values to get samples for the optimization algorithm
    for i in range(3):
        time = square_room.simulate()
        optimizing_algorithm.update_samples(square_room.heater_placement, scale_time(time))

    # Run until we get convergence
    while True:
        heater_placement = optimizing_algorithm.propose_location()
        if optimizing_algorithm.check_convergence():
            break
        time = square_room.simulate(heater_placement)
        optimizing_algorithm.update_samples(square_room.heater_placement, scale_time(time))

    return optimizing_algorithm.best_xy, optimizing_algorithm.best_t


if __name__ == '__main__':
    print(start_optimization())
