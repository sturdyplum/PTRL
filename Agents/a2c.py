import torch

class A2C: 
    def __init__(self, network, optimizer, value_coeff, entropy_coeff, max_grad_norm, writer):
        self.network = network
        self.optimizer = optimizer
        self.value_coeff = value_coeff
        self.entropy_coeff = entropy_coeff
        self.max_grad_norm = max_grad_norm
        self.mean = 0
        self.std = 1
        self.writer = writer
        # self.scheduler = scheduler

    def forward(self, observations, hidden_states):
        return self.network(observations, hidden_states)

    def select_actions(self, dist):
        return self.network.select_actions(dist)

    def normalize_actions(self, actions):
        return self.network.normalize_actions(actions)

    def updateModel(self, values, rewards, log_prob, advantages, entropy, new_std, new_mean, epoch):

        entropy_loss = self.entropy_coeff * (-entropy).mean()
        normalized_rewards = ((rewards - self.mean) / self.std).detach()
        value_loss = self.value_coeff * (values - normalized_rewards).pow(2).mean()
        
        policy_loss = (-log_prob * advantages.detach()).mean()

        self.writer.add_scalar("Value", value_loss, epoch)
        self.writer.add_scalar("Policy", policy_loss, epoch)
        self.writer.add_scalar("Entropy", entropy_loss, epoch)
        self.writer.add_scalar("Running_Mean_Reward", new_mean, epoch)
        self.writer.flush()

        a2c_loss = policy_loss + self.value_coeff * value_loss + self.entropy_coeff * entropy_loss
        # a2c_loss = value_loss + entropy_loss * self.entropy_coeff
        self.optimizer.zero_grad()
        a2c_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
        self.optimizer.step()
        # self.scheduler.step()

        # Update value layer with new mean and std for scale invariant updates.
        with torch.no_grad():
            self.network.value.weight *= self.std / new_std
            self.network.value.bias *= self.std
            self.network.value.bias += self.mean - new_mean
            self.network.value.bias /= new_std
        self.std = new_std
        self.mean = new_mean
