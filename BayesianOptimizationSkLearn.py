import numpy as np
import heat_eqn_2d as HE
import drift_diffusion_2d as DD
from skopt import gp_minimize
from skopt.plots import plot_convergence
import matplotlib.pyplot as plt
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
#square_room = HE.GridModel2D(parameters)
square_room = DD.GridModel2D_DD(parameters)


def f(x):
    value = (x[0] - 2) ** 2 + (x[1] - 2) ** 2
    return value


def func(x):
    value = square_room.simulate(x)
    return value/10000000


def optimize():
    result = gp_minimize(func,                  # the function to minimize
                         [(0, parameters['simulation']['Nx']-1), (0, parameters['simulation']['Nx']-1)], #  the bounds on each dimension of x
                         acq_func="EI",      # the acquisition function
                         n_calls=50,         # the number of evaluations of f
                         n_random_starts=5,  # the number of random initialization points
                         noise=10**(-2),     # the noise level (optional)
                         random_state=None)
    return result

"""
res_fun = []

for i in range(1):
    print("Iteration ", i)
    res_fun.append(optimize().fun)
print(sum(res_fun)/10)
print(max(res_fun))
print(min(res_fun))

plt.figure()
plt.plot(np.linspace(0, len(res_fun), len(res_fun)), res_fun)
plt.show()
"""

res = optimize()
plt.figure()
plt.plot(np.linspace(0, len(res.func_vals), len(res.func_vals)), res.func_vals)
plt.show()
