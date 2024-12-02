import torch
from torch.distributions.normal import Normal
from tqdm import tqdm  # for progress bar

class BlahutArimoto:
    def __init__(self, A=1, sigma=1, max_iter=10000, NX=500, NY=1000, tolerance=1e-6, epsilon=1e-12,
                 printInit=False, earlyStop=True, device=None):
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
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")

        if self.printInit:
            print(f"Device: {self.device}")
            print(f"A = {self.A}, sigma = {self.sigma}, max_iter = {self.max_iter}, NX = {self.NX}, NY = {self.NY}")
            print(f"tolerance = {self.tolerance}, epsilon = {self.epsilon}")

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
            p_y_given_x (2D tensor): Transition probability matrix P(y|x).
        """
        diff_x_y = (self.x.view(1, -1) - self.y.view(-1, 1))  # Compute pairwise differences

        # Compute half-width of interval (dy)
        dy = (self.y[1] - self.y[0]) * 0.5 if self.y.size(0) > 1 else 0.5  # Handle single-element case

        # Compute probabilities using the standard normal CDF
        normal_dist = Normal(0, self.sigma)
        p_y_given_x = normal_dist.cdf(diff_x_y + dy) - normal_dist.cdf(diff_x_y - dy)

        # Normalize columns to sum to 1 and avoid numerical instability
        p_y_given_x = torch.clamp(p_y_given_x, min=self.epsilon)
        p_y_given_x /= p_y_given_x.sum(dim=0, keepdim=True)

        self.p_y_given_x = p_y_given_x.to(self.device)

    def computeChannelCapacity(self):
        log_ratio = torch.log(torch.clamp(self.p_y_given_x / self.p_y.view(-1, 1), min=self.epsilon))
        self.capacity = torch.sum(self.p_x.flatten() * torch.sum(self.p_y_given_x * log_ratio, dim=0)).item()

    @staticmethod
    def theoreticalCapacity(P=1, _sigma=1):
        return 0.5 * torch.log2(1 + torch.tensor(P / _sigma**2)).item()

    def getMeanPower(self):
        self.mean_power = torch.sum(self.p_x * (self.x**2)).item()
        return self.mean_power

    def clearRecords(self):
        self.p_x_records = []
        self.p_y_records = []
        self.iter = None
        self.capacity = None
        self.theoretical_c = None
        self.mean_power = None

    def runAlgorithm(self, recordFrequency=50):
        self.clearRecords()
        # Discretize the input and output alphabets
        self.x = torch.linspace(-self.A, self.A, self.NX, device=self.device)
        self.y = torch.linspace(-self.A - 4 * self.sigma, self.A + 4 * self.sigma, self.NY, device=self.device)
        self.p_x = torch.ones((self.NX, 1), device=self.device) / self.NX  # Step 1 - Uniform initial input distribution

        self.computeChannelMatrix()  # Transition probabilities P(y|x)

        if self.printInit:
            print(f"Running on device: {self.device}")
            print(f"A = {self.A}, sigma = {self.sigma}, max_iter = {self.max_iter}, NX = {self.NX}, NY = {self.NY}")
            print(f"Initial input distribution p(x): max= {self.p_x.max()} min = {self.p_x.min()}")
            print(f"Transition probabilities p(y|x): max = {self.p_y_given_x.max()} min = {self.p_y_given_x.min()}")

        # Blahut-Arimoto iterations
        for iter in tqdm(range(self.max_iter), desc="Iterations", ncols=50):
            # Compute P(y)
            self.p_y = torch.matmul(self.p_y_given_x, self.p_x)  # Add epsilon for numerical stability
            self.p_y /= torch.sum(self.p_y)  # Normalize

            # Update q(x|y)
            self.q_x_given_y = (self.p_y_given_x * self.p_x.T).T / self.p_y.T
            self.q_x_given_y /= self.q_x_given_y.sum(dim=0, keepdim=True)  # Normalize rows

            # Update p(x)
            p_x_new = torch.exp(
                torch.sum(self.p_y_given_x * torch.log(torch.clamp(self.q_x_given_y.T, min=self.epsilon)), dim=0)
            )
            p_x_new /= torch.sum(p_x_new)  # Normalize

            # Check for convergence
            if self.earlyStop and torch.max(torch.abs(p_x_new - self.p_x)) < self.tolerance:
                print(f"Converged in {iter + 1} iterations.")
                break

            self.p_x = p_x_new

            # Record intermediate results
            if iter % recordFrequency == 0:
                self.p_x_records.append(self.p_x.clone())
                self.p_y_records.append(self.p_y.clone())

        self.computeChannelCapacity()
        self.getMeanPower()



        return {
            "A": self.A,
            "capacity": self.capacity,
            "theoretical_c": self.theoreticalCapacity(self.A, self.sigma),
            "x": self.x.cpu(),
            "y": self.y.cpu(),
            "p_x": self.p_x.cpu(),
            "p_y": self.p_y.cpu(),
            "q_x_given_y": self.q_x_given_y.cpu(),
            "p_y_given_x": self.p_y_given_x.cpu(),
            "p_x_records": [record.cpu() for record in self.p_x_records],
            "p_y_records": [record.cpu() for record in self.p_y_records],
            "iter": iter + 1,
            "mean_power": self.mean_power,
        }
