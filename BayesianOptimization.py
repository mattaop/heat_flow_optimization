import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.spatial.distance import cdist
from scipy.stats import norm


class BayesianOptimization:
    def __init__(self, parameters):
        self.xy_samples = np.empty((0, 2), int)
        self.t_samples = np.empty((0, 0), int)
        self.dim = self.xy_samples.shape
        self.bounds = np.array([[0, parameters['simulation']['Nx']-1], [0, parameters['simulation']['Ny']-1]])
        self.threshold = 10**(-3)

        self.convergence = False

        # Save best values
        self.best_xy = [0, 0]
        self.best_t = np.inf

    def _exponentiated_quadratic(self, xa, xb):
        """Exponentiated quadratic  with σ=1"""
        # L2 distance (Squared Euclidian)
        sq_norm = -1/25 * cdist(xa, xb, 'sqeuclidean')
        return np.exp(sq_norm)

    def _gaussian_process(self, x1, y1, x):
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
        mu, sigma = self._gaussian_process(self.xy_samples, self.t_samples, x)
        mu_sample_opt = np.max(self.t_samples)

        with np.errstate(divide='warn'):
            imp = mu - mu_sample_opt - xi
            z = imp / sigma
            ei = imp * norm.cdf(z) + sigma * norm.pdf(z)
            ei[sigma == 0.0] = 0.0
        return ei, mu, sigma

    def propose_location(self):
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
        #plt.show()

        plt.imshow(sigma_matrix)
        plt.colorbar()
        #plt.show()

        plt.imshow(ei_matrix)
        plt.colorbar()
        #plt.show()
        print(ei_matrix.max())
        if ei_matrix.max() <= self.threshold:
            print("Convergence")
            self.convergence = True
        return [i, j]

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

    def check_convergence(self):
        """
        Checks if the max expected improvement is less than the threshold
        :return: bool: If converged or not
        """
        return self.convergence
