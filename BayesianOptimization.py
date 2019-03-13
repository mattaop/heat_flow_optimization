import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern


class BayesianOptimization:
    def __init__(self, parameters):
        self.xy_samples = np.empty((0, 2), int)
        self.t_samples = np.empty((0, 0), int)
        self.dim = self.xy_samples.shape
        self.bounds = np.array([[0, parameters['simulation']['Nx']-1], [0, parameters['simulation']['Ny']-1]])
        self.noise = 10**(-10)
        self.threshold = np.maximum(parameters['simulation']["xlen"]/parameters['simulation']['Nx'],
                                    parameters['simulation']["ylen"]/parameters['simulation']['Ny'])

        # Setting kernel and Gaussian Process Regression
        self.m52 = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5)
        self.gpr = GaussianProcessRegressor(kernel=self.m52, normalize_y=True, alpha=self.noise ** 2)

        # Save best values
        self.best_xy = [0, 0]
        self.best_t = np.inf

    def _expected_improvement(self, x, xi=0.01):
        mu, sigma = self.gpr.predict(x, return_std=True)
        mu_sample = self.gpr.predict(self.xy_samples)
        #sigma = sigma.reshape(-1, self.xy_samples.shape[1])

        # Needed for noise-based model, otherwise use np.max(Y_sample).
        mu_sample_opt = np.max(mu_sample)

        with np.errstate(divide='warn'):
            imp = mu - mu_sample_opt - xi
            z = imp / sigma
            ei = imp * norm.cdf(z) + sigma * norm.pdf(z)
            ei[sigma == 0.0] = 0.0
        return ei

    def _propose_location(self, n_restarts=25):
        # Find the best optimum by starting from n_restart different random points.
        dim = self.xy_samples.shape[1]
        x0s = np.array(np.random.uniform(self.bounds[0, 0], self.bounds[0, 1], size=(n_restarts, dim)))

        def min_obj(X):
            # Minimization objective is the negative acquisition function
            return -self._expected_improvement(X.reshape(-1, dim))

        for x0 in x0s:  # To avoid local minima
            res = minimize(min_obj, x0=x0, bounds=self.bounds, method='L-BFGS-B')
            if res.fun < self.best_t:  # If value of new point is smaller than min_val, update min_val
                self.best_t = res.fun[0]
                self.best_xy = res.x
        return self.best_xy

    def bayesian_optimization(self):
        """
        Finds next coordinate to simulate
        :return: coordinate to simulate
        """
        # Update Gaussian process with existing samples
        self.gpr.fit(self.xy_samples, self.t_samples)

        # Obtain next sampling point from the acquisition function (expected_improvement)
        xy_next = self._propose_location()

        # Return ints for placement in array
        xy_next = np.round(xy_next, 0)
        xy_next = xy_next.astype(int)

        # Obtain next noisy sample from the objective function
        return xy_next

    def update_samples(self, new_xy_sample, new_t_sample):
        """
        Update samples
        :param new_xy_sample: Coordinates of latest sample to be added
        :param new_t_sample: Temperature of tha latest sample to be added
        :return: none
        """
        self.xy_samples = np.append(self.xy_samples, [new_xy_sample], axis=0)
        self.t_samples = np.append(self.t_samples, [new_t_sample])
        """
        print(self.t_samples)
        if self.xy_samples.size < 2:
            self.xy_samples = np.array([new_xy_sample])
        else:
            self.xy_samples = np.concatenate((self.xy_samples, [new_xy_sample]))
        if self.t_samples:
            self.t_samples = np.concatenate((self.t_samples, [new_t_sample]))
        else:
            self.t_samples = [new_t_sample]
        """
        self.dim = self.xy_samples.shape

    def convergence(self):
        if np.amax(self.xy_samples[-1, :]-self.xy_samples[-3:, :]) <= self.threshold:
            return True
        else:
            return False
