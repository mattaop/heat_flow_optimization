import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.spatial.distance import cdist
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
        self.noise = 10**(-2)
        self.threshold = 1

        # Setting kernel and Gaussian Process Regression
        self.m52 = ConstantKernel(2.0) * Matern(length_scale=2.0, nu=2.5)
        self.gpr = GaussianProcessRegressor(kernel=self.m52, normalize_y=True, alpha=self.noise ** 2)


        # Save best values
        self.best_xy = [0, 0]
        self.best_t = np.inf

    def _exponentiated_quadratic(self, xa, xb):
        """Exponentiated quadratic  with σ=1"""
        # L2 distance (Squared Euclidian)
        sq_norm = -1/25 * cdist(xa, xb, 'sqeuclidean')
        return np.exp(sq_norm)

    def _GP(self, x1, y1, x):
        """
        Calculate the posterior mean and covariance matrix for y2
        based on the corresponding input X2, the observations (y1, X1),
        and the prior kernel function.
        """
        # Kernel of the observations
        sigma_11 = self._exponentiated_quadratic(x1, x1)
        for i in range(self.bounds[0, 1]):
            for j in range(self.bounds[1, 1]):
                # Kernel of observations vs to-predict
                sigma_12 = self._exponentiated_quadratic(x1, x)
                # Solve
                solved = scipy.linalg.solve(sigma_11, sigma_12, assume_a='pos').T
                # Compute posterior mean
                mu_2 = solved @ y1
                # Compute the posterior covariance
                sigma_22 = self._exponentiated_quadratic(x, x)
                sigma_2 = sigma_22 - (solved @ sigma_12)
        return mu_2, sigma_2

    def _expected_improvement(self, x, xi=0.01):
        #print("x", x)
        #print("earlier samples", self.xy_samples)

        mu, sigma = self._GP(self.xy_samples, self.t_samples, x)
        #print(sigma)
        # sigma = sigma.reshape(-1, self.xy_samples[1])

        # Needed for noise-based model, otherwise use np.max(Y_sample).
        mu_sample_opt = np.max(self.t_samples)

        with np.errstate(divide='warn'):
            imp = mu - mu_sample_opt - xi
            z = imp / sigma
            ei = imp * norm.cdf(z) + sigma * norm.pdf(z)
            ei[sigma == 0.0] = 0.0
        return ei, mu, sigma

    def _propose_location(self, n_restarts=25):
        # Find the best optimum by starting from n_restart different random points.
        dim = self.xy_samples.shape[1]
        x0s = np.array(np.random.uniform(self.bounds[0, 0], self.bounds[0, 1], size=(n_restarts, dim)))
        min_val = np.inf
        min_x = None

        def min_obj(x):
            return self._expected_improvement(x.reshape(-1, dim))

        for x0 in x0s:  # To avoid local minima
            #print("Propose: ", x0)
            res = minimize(min_obj, x0=x0, bounds=self.bounds, method='L-BFGS-B')
            #print("Best", res.x)
            print(res.fun, x0, res.x)
            if res.fun < min_val:  # If value of new point is smaller than min_val, update min_val
                min_val = res.fun[0]
                min_x = res.x
        print("min_x", min_x)
        return min_x

    def _propose_location_2(self):
        dim = self.xy_samples.shape[1]
        ei_matrix = np.zeros([self.bounds[0, 1], self.bounds[1, 1]])
        mu_matrix = np.zeros([self.bounds[0, 1], self.bounds[1, 1]])
        sigma_matrix = np.zeros([self.bounds[0, 1], self.bounds[1, 1]])
        for i in range(self.bounds[0, 1]):
            for j in range(self.bounds[1, 1]):
                ei_matrix[i, j], mu_matrix[i, j], sigma_matrix[i, j] = self._expected_improvement(np.array([i, j]).reshape(-1, dim))
        i, j = np.unravel_index(ei_matrix.argmax(), ei_matrix.shape)

        plt.imshow(mu_matrix)
        plt.colorbar()
        plt.show()

        plt.imshow(sigma_matrix)
        plt.colorbar()
        plt.show()

        plt.imshow(ei_matrix)
        plt.colorbar()
        plt.show()
        return [i, j]

    def optimization(self):
        """
        Finds next coordinate to simulate
        :return: coordinate to simulate
        """
        # Update Gaussian process with existing samples
        self.gpr.fit(self.xy_samples, self.t_samples)

        # Obtain next sampling point from the acquisition function (expected_improvement)
        xy_next = self._propose_location_2()

        # Return ints for placement in array
        #xy_next = np.round(xy_next, 0)
        #xy_next = xy_next.astype(int)

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
        self.best_t = np.max(self.t_samples)
        self.best_xy = self.xy_samples[np.argmax(self.t_samples)]
        self.dim = self.xy_samples.shape

    def convergence(self):
        if np.abs(self.t_samples[-1]-np.mean([self.t_samples[-2], self.t_samples[-3]])) <= self.threshold:
            return True
        else:
            return False
