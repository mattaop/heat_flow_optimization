from simulation import drift_diffusion_2d as DD, heat_eqn_2d as HE
import numpy as np
import time
import json
from optimization_alogrithms import GradientDescent as GD, BayesianOptimization as BO

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
    """Log'ing the time, and changing sign, since we ar minimizing."""
    return -np.log(time)


def start_optimization(algorithm, input_file, model):
    """
    Run the optimization algorithm specified.
    :param algorithm: Algorithm to be used for optimization ('GD' or 'BO')
    :param input_file: path to .json file with input data
    :param model: Heat model for simulations ('DD' or 'HE')
    :return: Optimal position, runtime for optimization and for simulation depending on algorithm
    """
    parameters = load_initial_values(input_file)
    if model == 'HE':
        square_room = HE.GridModel2D(parameters)
    if model == 'DD':
        square_room = DD.GridModel2D_DD(parameters)

    # Bayesian optimization
    if algorithm == 'BO':
        optimizing_algorithm = BO.BayesianOptimization(parameters)
        optimization_time = 0
        simulation_time = 0

        # Run the simulation for two random values to get samples for the optimization algorithm
        for i in range(3):
            #print('Simulate location', len(optimizing_algorithm.t_samples)+1, '...')
            start_time = time.time()
            time_at_placement = square_room.simulate(velocity_field = 'directional')
            simulation_time += time.time()-start_time

            start_time = time.time()
            optimizing_algorithm.update_samples(square_room.heater_placement, scale_time(time_at_placement))
            optimization_time += time.time()-start_time
        # square_room.plot_temperature_room()

        # Run until we get convergence
        start_time = time.time()
        while True:
            heater_placement = optimizing_algorithm.propose_location(plot=None)
            #print("Expected improvement left: ", optimizing_algorithm.ei)
            #print("Threshold for convergence: ", optimizing_algorithm.threshold)
            if optimizing_algorithm.check_convergence():
                break
            optimization_time += time.time()-start_time
            start_time = time.time()
            #print('Simulate location', len(optimizing_algorithm.t_samples)+1, '...')
            time_at_placement = square_room.simulate(heater_placement = heater_placement, velocity_field = 'directional')
            simulation_time += time.time()-start_time
            start_time = time.time()
            optimizing_algorithm.update_samples(square_room.heater_placement, scale_time(time_at_placement))
            print(optimizing_algorithm.best_xy)
        return len(optimizing_algorithm.t_samples+1), optimization_time, simulation_time

    # Gradient descent-based optimization
    if algorithm == 'GD':
        optimizer = GD.GradientDescent(parameters, model)
        return optimizer.optimize(k=1)


def main():
    """Run both the Bayesian optimization and the gradient descent based optimization"""
    number_of_trials = 5  # How many times to run each optimization algorithm

    print("Running Bayesian optimization", number_of_trials, "time(s)")
    number_of_iterations = np.zeros([number_of_trials])
    time_per_iteration = np.zeros([number_of_trials])
    optimization_time = np.zeros([number_of_trials])
    simulation_time = np.zeros([number_of_trials])
    for i in range(number_of_trials):
        print('Run ', i)
        start_time = time.time()
        number_of_iterations[i], optimization_time[i], simulation_time[i] = start_optimization('BO',
                                                                                               'initial_values/rectangular.json', 'DD' )
        time_per_iteration[i] = time.time()-start_time
    print('Average number of iterations over ', number_of_trials, ' trials: ', number_of_iterations.mean())
    print('Average time over ', number_of_trials, 'trials: ', time_per_iteration.mean(), 's , with optimization: ',
          optimization_time.mean(), ' s, and simulation: ', simulation_time.mean(), ' s.')

    print("Running gradient descent-based optimization", number_of_trials, "time(s)")
    positions = []
    steps = []
    times = []
    opt_times = []
    sim_times = []
    for i in range(number_of_trials):
        start_time = time.time()
        print('Run ', i)
        pos, number_steps, optimization_time, simulation_time = start_optimization('GD',
                                                                                   'initial_values/rectangular.json', 'DD')
        positions.append(pos)
        steps.append(number_steps)
        times.append(time.time() - start_time)
        opt_times.append(optimization_time)
        sim_times.append(simulation_time)
    print('Average number of iterations over ', number_of_trials, ' trials: ', np.mean(steps))
    print('Average time over ', number_of_trials, 'trials: ', np.mean(times), 's , with optimization: ',
          np.mean(opt_times), ' s, and simulation: ', np.mean(sim_times), ' s.')
    print('Optimal positions:', positions)


if __name__ == '__main__':
    main()
