import BayesianOptimization as BO
import heat_eqn_2d as HE
import drift_diffusion_2d as DD
import test
import numpy as np
import random
import time
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
    return -np.log(time)


def start_optimization():
    parameters = load_initial_values("initial_values/mathias.json")

    heater_placement = [0, 0]
    #square_room = HE.GridModel2D(parameters)
    square_room = DD.GridModel2D_DD(parameters)
    optimizing_algorithm = BO.BayesianOptimization(parameters)
    test_function = test.TestFunction()
    optimization_time = 0
    simulation_time = 0

    # Run the simulation for two random values to get samples for the optimization algorithm
    for i in range(10):
        print('Simulate location', len(optimizing_algorithm.t_samples)+1, '...')
        start_time = time.time()
        time_at_placement = square_room.simulate()
        simulation_time += time.time()-start_time
        
        start_time = time.time()
        optimizing_algorithm.update_samples(square_room.heater_placement, scale_time(time_at_placement))
        optimization_time += time.time()-start_time
    # square_room.plot_temperature_room()

    # Run until we get convergence
    start_time = time.time()
    while True:
        heater_placement = optimizing_algorithm.propose_location()
        print("Expected improvement left: ", optimizing_algorithm.ei)
        print("Threshold for convergence: ", optimizing_algorithm.threshold)
        if optimizing_algorithm.check_convergence():
            break
        optimization_time += time.time()-start_time
        start_time = time.time()
        print('Simulate location', len(optimizing_algorithm.t_samples)+1, '...')
        time_at_placement = square_room.simulate(heater_placement)
        simulation_time += time.time()-start_time
        start_time = time.time()
        optimizing_algorithm.update_samples(square_room.heater_placement, scale_time(time_at_placement))
    return len(optimizing_algorithm.t_samples+1), optimization_time, simulation_time


if __name__ == '__main__':
    number_of_trials = 1
    number_of_iterations = np.zeros([number_of_trials])
    time_per_iteration = np.zeros([number_of_trials])
    optimization_time = np.zeros([number_of_trials])
    simulation_time = np.zeros([number_of_trials])
    for i in range(number_of_trials):
        start_time = time.time()
        number_of_iterations[i], optimization_time[i], simulation_time[i] = start_optimization()
        time_per_iteration[i] = time.time()-start_time
    print('Average number of iterations over ', number_of_trials, ' trials: ', number_of_iterations.mean())
    print('Average time over ', number_of_trials, 'trials: ', time_per_iteration.mean(), 's , with optimization: ',
          optimization_time.mean(), ' s, and simulation: ', simulation_time.mean(), ' s.')
