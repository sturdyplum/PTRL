import time
import torch

class RenderAgent:
    def render(self, environment, network, num_hidden_states, queue, sleep_time=.01):
        observation = environment.reset()
        game = -1
        while True:
            game += 1
            while not queue.empty():
                network = queue.get()
            reward = 0
            steps = 0
            hidden_state = torch.zeros((1, 1, num_hidden_states), dtype=torch.float)
            while True:
                steps += 1
                policy, _, hidden_state = network.forward(torch.tensor([observation], dtype=torch.float), hidden_state)
                environment.render()
                time.sleep(sleep_time)
                action = network.select_actions(policy)
                observation, r, done, _ = environment.step(action[0].detach().numpy())
                reward += r
                if done:
                    observation = environment.reset()
                    break