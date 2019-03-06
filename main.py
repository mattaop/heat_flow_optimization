import BayesianOptimization as BO
import heat_eqn_2d as HE
import json


print("Project in modelling and optimization of heat flow")


def load_initial_values(input_file):
    """
    Setting initial values for the optimization
    :param inputfile: JSON file containing input data
    :return:
    """
    data = json.load(open(input_file, "r"))
    print(data)
    return data


def start_optimization():
    parameters = load_initial_values("initial_values/mathias.json")
    xlen = parameters["simulation"]["xlen"]
    ylen = parameters["simulation"]["ylen"]
    Nx = parameters["simulation"]["Nx"]
    Ny = parameters["simulation"]["Ny"]
    initial_temperature = parameters["simulation"]["initial_temperature"]
    heater_temperature = parameters["simulation"]["heater_temperature"]
    outside_temperature = parameters["simulation"]["outside_temperature"]
    temperature_goal = parameters["simulation"]["temperature_goal"]
    heater_placement = (0, 0)
    square_room = HE.GridModel2D(xlen, ylen, Nx, Ny, initial_temperature, heater_temperature, outside_temperature, temperature_goal)
    xy_samples = parameters["optimization"]["xy_samples"]
    t_samples = parameters["optimization"]["t_samples"]
    bounds = parameters["optimization"]["bounds"]
    optimizing_algorithm = BO.BayesianOptimization(xy_samples, t_samples, bounds)
    # Run the simulation for two random values to get samples for the optimization algorithm
    for i in range(2):
        time = square_room.simulate()
        optimizing_algorithm.update_samples(square_room.heater_placement, time)
    # Run until we get a optimal solution
    optimizing = True
    while optimizing:
        heater_placement = optimizing_algorithm.bayesian_optimization()
        time = square_room.simulate(heater_placement)
        optimizing_algorithm.update_samples(square_room.heater_placement, time)

        # For some condition to stop optimizing:
            # Break
    return heater_placement


if __name__ == '__main__':
    start_optimization()