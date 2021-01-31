class RewardsNormalizer:
    def __init__(self, decay):
        self.running_mean = 0
        self.varience = 1
        self.decay = decay

    def add_variables(self, x):
        for var in x:
            self.add_variable(var.item())

    def add_variable(self, x):
        self.running_mean = (1.0 - self.decay) * self.running_mean + self.decay  * x
        self.varience = (1.0 - self.decay) * self.varience + self.decay * ((x-self.running_mean) ** 2)

    def mean(self):
        return self.running_mean
        # return 0

    def std(self):
        return max((self.varience)**.5, 1e-5)
        # return 1