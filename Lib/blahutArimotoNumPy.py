import numpy as np
from scipy.stats import norm
from tqdm import tqdm # for progress bar
from datetime import datetime

# Class implementation
class BlahutArimoto:
    def __init__(self, A=1, sigma=1, max_iter=1000, NX=500, NY=1000, tolerance=1e-6, epsilon=1e-12,
                 printInit=False, earlyStop=True):
        # Default variables
        self.A = A
        self.sigma = sigma
        self.max_iter = max_iter
        self.NX = NX
        self.NY = NY
        self.tolerance = tolerance
        self.epsilon = epsilon
        self.printInit = printInit
        self.earlyStop = earlyStop

        # To be computed
        self.x = None
        self.y = None
        self.p_x = None
        self.p_y = None
        self.q_x_given_y = None
        self.p_y_given_x = None
        self.p_x_records = []
        self.p_y_records = []
        self.iter = None
        self.capacity = None
        self.theoretical_c = None
        self.mean_power = None
        self.recordFrequency = 50

        self.result = None

    def computeChannelMatrix(self):
        """
        Compute the transition probabilities P(y|x) for a Gaussian noise channel.
        Returns:
            p_y_given_x (2D array): Transition probability matrix P(y|x).
        """
        # Compute pairwise differences
        diff_x_y = ((self.x).reshape(1, -1) - (self.y).reshape(-1, 1))

        # Compute half-width of interval (dy)
        dy = (self.y[1] - self.y[0]) * 0.5 if len(y) > 1 else 0.5  # Handle single-element case

        # Compute probabilities using the standard normal CDF
        p_y_given_x = norm.cdf(diff_x_y + dy, scale=self.sigma) - norm.cdf(diff_x_y - dy, scale=self.sigma)

        # Normalize columns to sum to 1
        self.p_y_given_x /= p_y_given_x.sum(axis=0, keepdims=True)

        self.p_y_given_x = np.maximum(p_y_given_x, self.epsilon)

        return self.p_y_given_x

    def computeChannelCapacity(self):
        log_ratio = np.log(np.maximum(self.p_y_given_x / self.p_y[:, None], self.epsilon))
        self.capacity = np.sum(self.p_x.flatten() * np.sum(self.p_y_given_x * log_ratio, axis=0))

    def theoreticalCapacity(P, _sigma=1):
        return 0.5 * np.log2(1 + P / _sigma**2)

    def getMeanPower(self):
        # Calculate mean power as E[X^2] = sum(p(x) * x^2)
        self.mean_power = np.sum(self.p_x * (self.x)**2)
        return self.mean_power

    def clearRecords(self):
        self.p_x_records = []
        self.p_y_records = []
        self.iter = None
        self.capacity = None
        self.theoretical_c = None
        self.mean_power = None

    def run(self, recordFrequency=50):
        self.clearRecords()
        # Discretize the input and output alphabets
        self.x = np.linspace(-self.A, self.A, self.NX)
        self.y = np.linspace(-self.A - 4 * self.sigma, self.A + 4 * self.sigma, self.NY)
        self.p_x = np.ones((self.NX, 1)) / self.NX                  # Step 1 - Uniform initial input distribution

        self.computeChannelMatrix() # Transition probabilities P(y|x)

        if self.printInit:
            print(f"A = {self.A}, sigma = {self.sigma}, max_iter = {self.max_iter}, NX = {self.NX}, NY = {self.NY}")
            print(f"Initial input distribution p(x): (size {self.p_x.shape}) max= {self.p_x.max()} min = {self.p_x.min()}")
            print(f"Input alphabet: (size = {self.x.shape}) max = {self.x.max()} min = {self.x.min()}")
            print(f"Output alphabet: (size = {self.y.shape}) max = {self.y.max()} min = {self.y.min()}")
            print(f"Transition probabilities p(y|x): (size {self.p_y_given_x.shape}) max = {self.p_y_given_x.max()} min = {self.p_y_given_x.min()}")

        # Blahut-Arimoto iterations
        for iter in tqdm(range(self.max_iter), desc="Iterations", ncols=50):
            # Compute P(y)
            self.p_y = np.dot(self.p_y_given_x, self.p_x)  # Add epsilon for numerical stability
            self.p_y /= np.sum(self.p_y)              # Normalize

            # Update q(x|y)
            self.q_x_given_y = (self.p_y_given_x * self.p_x.reshape(1, -1)).T / self.p_y.T  # Shape (NX, NY)
            self.q_x_given_y /= self.q_x_given_y.sum(axis=0, keepdims=True)  # Normalize rows

            # Update p(x)
            p_x_new = np.exp(np.sum(self.p_y_given_x * np.log(np.maximum(self.q_x_given_y.T, self.epsilon)), axis=0))
            p_x_new /= np.sum(p_x_new)  # Normalize

            # Check for convergence
            if self.earlyStop and np.max(np.abs(p_x_new - self.p_x)) < self.tolerance:
                print(f"Converged in {iter + 1} iterations.")
                break

            self.p_x = p_x_new

            # Record intermediate results
            if iter % recordFrequency == 0:
                self.p_x_records.append(p_x_new.copy())
                self.p_y_records.append(self.p_y.copy())

        self.computeChannelCapacity()
        self.getMeanPower()

        return {
            "A": self.A,
            "capacity": self.capacity,
            "theoretical_c": self.theoreticalCapacity(self.A, self.sigma),
            "x": self.x,
            "y": self.y,
            "p_x": self.p_x,
            "p_y": self.p_y,
            "q_x_given_y": self.q_x_given_y,
            "p_y_given_x": self.p_y_given_x,
            "p_x_records": self.p_x_records,
            "p_y_records": self.p_y_records,
            "iter": self.iter + 1,
            "mean_power": self.mean_power
      }
    
    def saveResult(self):
        return {
            "A": self.A,
            "capacity": self.capacity,
            "theoretical_c": self.theoreticalCapacity(self.A, self.sigma),
            "x": self.x,
            "y": self.y,
            "p_x": self.p_x,
            "p_y": self.p_y,
            "q_x_given_y": self.q_x_given_y,
            "p_y_given_x": self.p_y_given_x,
            "p_x_records": self.p_x_records,
            "p_y_records": self.p_y_records,
            "iter": self.iter + 1,
            "mean_power": self.mean_power
      }

