import BayesianOptimization as BO
import heat_eqn_2d as HE
import json


print("Project in modelling and optimization of heat flow")


def load_initial_values():
    """
    Setting initial values for the optimization
    :return:
    """
    with open("initial_values\mathias.json", "r") as read_file:
        data = json.load(read_file)
    print(data)
    return data


def start_optimization():
    initial_values = load_initial_values()

    heater_placement = (0, 0)
    square_room = HE.GridModel2D()
    optimizing_algorithm = BO.BayesianOptimization()
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
