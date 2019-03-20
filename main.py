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


def start_optimization():
    parameters = load_initial_values("initial_values/mathias.json")

    heater_placement = [0, 0]
    square_room = HE.GridModel2D(parameters)
    optimizing_algorithm = BO.BayesianOptimization(parameters)
    test_function = test.TestFunction()

    # Run the simulation for two random values to get samples for the optimization algorithm
    for i in range(3):
        # time = square_room.simulate()
        time = test_function.f([random.randint(0, 9), random.randint(0, 9)])
        # print(square_room.heater_placement, time)
        optimizing_algorithm.update_samples(heater_placement, time)

    # Run until we get convergence
    # while not optimizing_algorithm.convergence():
    for i in range(5):
        heater_placement = optimizing_algorithm.bayesian_optimization()
        # time = square_room.simulate(heater_placement)
        time = test_function.f(heater_placement)
        # print(heater_placement, time)
        # optimizing_algorithm.update_samples(square_room.heater_placement, time)
        optimizing_algorithm.update_samples((heater_placement[0], heater_placement[1]), time)

    return optimizing_algorithm.best_xy, optimizing_algorithm.best_t


if __name__ == '__main__':
    print(start_optimization())
