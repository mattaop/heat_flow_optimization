import numpy as np
import heat_eqn_2d as HE
from skopt import gp_minimize
import json


def load_initial_values(input_file):
    """
    Setting initial values for the optimization
    :param input_file: Path to json file
    :return: Dict containing data from input_file
    """

    data = json.load(open(input_file, "r"))
    return data


parameters = load_initial_values("initial_values/mathias.json")
square_room = HE.GridModel2D(parameters)


def f(x):
    print(x)
    return square_room.simulate(x)


res = gp_minimize(f,                  # the function to minimize
                  [(0, parameters['simulation']['Nx']-1), (0, parameters['simulation']['Nx']-1)],   # the bounds on each dimension of x
                  acq_func="EI",      # the acquisition function
                  n_calls=20,         # the number of evaluations of f
                  n_random_starts=3,  # the number of random initialization points
                  noise=0.1**10       # the noise level (optional)
                  random_state=None)
print(res.x)
print(f([5, 5]))
