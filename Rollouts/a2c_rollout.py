import torch

def discount(arr, mask, discount_rate, end_val):
    """
    Returns a discounted suffix sum of `arr` to elements of the array
    along the last dimention.
    `arr`: Array to discount. Not modified but copied.
    `mask`: Array that determins when to cut off the suffix sum.
    `discount_rate`: discount to apply when summing.
    """
    arr_copy = arr.clone()
    for i in reversed(range(len(arr_copy))):
        if i == len(arr_copy) - 1: 
            arr_copy[i] += (1-mask[i]) * discount_rate * end_val
        else:
            arr_copy[i] += arr_copy[i+1] * (1-mask[i]) * discount_rate
    return arr_copy

class RolloutResult:

    def __init__(self, batch_size, log_probs, advantage, values, rewards, entropy):
        self.log_probs = log_probs.view(batch_size)
        self.advantage = advantage.view(batch_size).detach()
        self.values = values.view(batch_size)
        self.rewards = rewards.view(batch_size).detach()
        self.entropy = entropy.view(batch_size)

class A2CRollout:
    """
    This class will store the rollout for an environment for a group of agents.
    Once all the rollout is collected, it will perform any preprocessing on the
    data to make it ready to be used for training.
    """
    def __init__(self, batch_size, num_agents, gamma, advantage_lambda):
        self.rewards = torch.empty((batch_size, num_agents), dtype=torch.float)
        self.log_prob = torch.empty((batch_size, num_agents), dtype=torch.float)
        self.entropy =  torch.empty((batch_size, num_agents), dtype=torch.float)
        self.values = torch.empty((batch_size, num_agents), dtype=torch.float)
        self.dones = torch.empty((batch_size, num_agents), dtype=torch.float)
        self.advantages = torch.empty((batch_size, num_agents), dtype=torch.float)
        self.step = 0
        self.batch_size = batch_size
        self.num_agents = num_agents
        self.gamma = gamma
        self.advantage_lambda = advantage_lambda
        
    def add(self,
            reward, 
            policy, 
            action_selected, 
            done, 
            value):
        """Adds a step of information for a group of environments."""
        if self.step == len(self.rewards):
            self.reset()
        self.rewards[self.step] = torch.tensor(reward, dtype=torch.float)
        # print("POLICY: " , policy.mean, ":", policy.stddev, " LOG PROBS: " , policy.log_prob(action_selected), " ACTIONS: " , action_selected)
        self.log_prob[self.step] = policy.log_prob(action_selected).sum(dim=-1)
        self.entropy[self.step] = policy.entropy().sum(dim=-1)
        self.values[self.step] = value
        self.dones[self.step] = torch.tensor(done, dtype=torch.float)
        self.step += 1

    def reset(self):
        self.rewards = torch.empty((self.batch_size, self.num_agents), dtype=torch.float)
        self.log_prob = torch.empty((self.batch_size, self.num_agents), dtype=torch.float)
        self.entropy =  torch.empty((self.batch_size, self.num_agents), dtype=torch.float)
        self.values = torch.empty((self.batch_size, self.num_agents), dtype=torch.float)
        self.dones = torch.empty((self.batch_size, self.num_agents), dtype=torch.float)
        self.step = 0

    def denorm(self, x, std, mean):
        return x * std + mean

    def norm(self, x, std, mean):
        return (x - mean) / std

    def do_discounts(self, final_rewards, std, mean):
        last_advantage = 0
        for i in reversed(range(len(self.rewards))):
            if i == len(self.rewards) - 1:
                last_values = final_rewards
            else:
                last_values = self.values[i+1]

            delta_t =  self.norm(self.rewards[i] + self.gamma * (1 - self.dones[i]) * 
                                 self.denorm(last_values, std, mean),
                                 std,mean) - self.values[i]
            
            last_advantage = delta_t + self.gamma * self.advantage_lambda * (1-self.dones[i]) * last_advantage
            self.advantages[i] = last_advantage.detach()

        self.rewards = self.denorm(self.values + self.advantages, std, mean).detach()


    def make_data(self, 
                  final_rewards: torch.FloatTensor, 
                  std: float,
                  mean: float):
        self.do_discounts(final_rewards, std, mean)
        new_batch_size = self.batch_size * self.num_agents
        return RolloutResult(new_batch_size, 
                            self.log_prob, 
                            self.advantages, 
                            self.values, 
                            self.rewards, 
                            self.entropy)
        

        