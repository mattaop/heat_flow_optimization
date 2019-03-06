import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern


class BayesianOptimization:
    def __init__(self, xy_samples, t_samples, bounds):
        self.xy_samples = np.array(xy_samples)
        self.t_samples = np.array(t_samples)
        self.dim = self.xy_samples.shape
        self.bounds = bounds
        self.noise = 10**(-10)

        # Setting kernel and Gaussian Process Regression
        self.m52 = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5)
        self.gpr = GaussianProcessRegressor(kernel=self.m52, alpha=self.noise ** 2)

        #
        self.best_xy = [0, 0]
        self.best_t = np.inf

    def expected_improvement(self, x, xi=0.01):
        mu, sigma = self.gpr.predict(x, return_std=True)
        mu_sample = self.gpr.predict(self.xy_samples)

        sigma = sigma.reshape(-1, self.xy_samples.shape[1])

        # Needed for noise-based model, otherwise use np.max(Y_sample).
        mu_sample_opt = np.max(mu_sample)

        with np.errstate(divide='warn'):
            imp = mu - mu_sample_opt - xi
            z = imp / sigma
            ei = imp * norm.cdf(z) + sigma * norm.pdf(z)
            ei[sigma == 0.0] = 0.0
        return ei

    def propose_location(self, n_restarts=25):
        # Find the best optimum by starting from n_restart different random points.
        for x0 in np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], size=(n_restarts, self.dim)): # To avoid local minima
            res = minimize(self.expected_improvement, x0=x0, bounds=self.bounds, method='L-BFGS-B')
            if res.fun < self.best_t: # If value of new point is smaller than min_val, update min_val
                self.best_t = res.fun[0]
                self.best_xy = res.x
        return self.best_xy.reshape(-1, 1)

    def bayesian_optimization(self):
        """
        Finds next coordinate to simulate
        :return: coordinate to simulate
        """
        # Update Gaussian process with existing samples
        self.gpr.fit(self.xy_samples, self.t_samples)

        # Obtain next sampling point from the acquisition function (expected_improvement)
        xy_next = self.propose_location()

        # Obtain next noisy sample from the objective function
        return xy_next

    def update_samples(self, new_xy_sample, new_t_sample):
        """
        Update samples
        :param new_xy_sample: Coordinates of latest sample to be added
        :param new_t_sample: Temperature of tha latest sample to be added
        :return: none
        """
        self.xy_samples = np.vstack([self.xy_samples, new_xy_sample])
        self.t_samples = np.vstack([self.t_samples, new_t_sample])
        self.dim = self.xy_samples.shape
