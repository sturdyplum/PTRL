from Runners.render import RenderAgent
from Networks.linear_network import LinearNet
from Networks.linear_network_cont import LinearNetCont

from Agents.a2c import A2C
from Environment.vectorized_gym import GymOnSubprocesses
from torch.utils.tensorboard import SummaryWriter
from queue import Queue
from Rollouts.a2c_rollout import A2CRollout
from Rollouts.running_stats import RewardsNormalizer

import torch 
import gym
import threading
import copy
import time

pop_art_decay = 3e-4
agent_type = "A2C"
network_type = "Linear-Cont"
environment_name = "BipedalWalker-v3"
num_epochs = 1000000
num_actions = 4
num_inputs = 24
learning_rate = 1e-3
adv_lambda = .95
gamma = .99
value_coeff = .5
entropy_coeff = .01
max_grad_norm = .5
layer_inputs = [num_inputs, 256, 128]
num_proccesses = 5
num_steps = 200
num_agents_per_process = 10
test_epoch = 4
should_render_agent = True
activation_function = torch.nn.functional.relu 
optimizer_type = "Adam"
num_hidden_state = 20

def main():
    environment_function = lambda : gym.make(environment_name)
    if network_type == "Linear":
        network = LinearNet(layer_inputs, num_actions, activation_function, num_hidden_state)
    elif network_type == "Linear-Cont":
        # This should only be for continous action spaces.
        env = gym.make(environment_name)
        network = LinearNetCont(layer_inputs, env.action_space, activation_function, num_hidden_state)
        env.close()
    else:
        raise NotImplementedError

    if optimizer_type == "RMSprop":
        optimizer = torch.optim.RMSprop(network.parameters(), lr=learning_rate)
    elif optimizer_type == "Adam":
        optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)
    else:
        raise NotImplementedError

    writer = SummaryWriter()
    if agent_type == "A2C":
        agent = A2C(network, optimizer, value_coeff, entropy_coeff, max_grad_norm, writer)
        rollout = A2CRollout(num_steps, num_agents_per_process * num_proccesses, gamma, adv_lambda)
    else:
        raise NotImplementedError

    if should_render_agent:
        render_agent = RenderAgent()
        render_queue = Queue()
        render = threading.Thread(target=render_agent.render,
                                  args=(environment_function(), 
                                        copy.deepcopy(agent.network),
                                        num_hidden_state, render_queue,  
                                        .01))
        render.setDaemon(True)
        render.start()
    
    environments = GymOnSubprocesses(environment_function, num_proccesses, num_agents_per_process)
    observations = torch.tensor(environments.reset(), dtype=torch.float)
    hidden_states = torch.zeros((1, num_proccesses * num_agents_per_process, num_hidden_state), dtype=torch.float)
    rewards_normalizer = RewardsNormalizer(pop_art_decay)

    infrence_time = 0
    envrironment_steps_time = 0
    rollout_time = 0
    normalizer_time = 0
    update_time = 0
    test_time = 0
    total_time = 1

    for epoch in range(num_epochs):

        print ("Current epoch: %d TIMES[infrence: %.2f steps: %.2f rollout: %.2f normalizer: %.2f updates: %.2f test: %.2f]                     " 
        %(epoch, 
          infrence_time/total_time, 
          envrironment_steps_time/total_time, 
          rollout_time /total_time, 
          normalizer_time / total_time, 
          update_time/total_time, 
          test_time/total_time), end='\r')

        for _ in range(num_steps):
            infrence_start = time.time()
            policy, value, hidden_states = agent.forward(observations, hidden_states)
            actions = agent.select_actions(policy)
            infrence_time += time.time() - infrence_start

            envrironment_steps_start = time.time()
            environments.step_async(actions)
            result = environments.step_wait()
            envrironment_steps_time += time.time() - envrironment_steps_start

            rollout_start = time.time()
            hidden_states = (hidden_states * (1 - torch.tensor(result.done, dtype=torch.float)).unsqueeze(1)).detach()
            observations = torch.tensor(result.observation, dtype=torch.float)
            rollout.add(
                copy.deepcopy(result.reward), 
                policy, 
                agent.normalize_actions(actions), 
                copy.deepcopy(result.done), 
                value)
            rollout_time += time.time() - rollout_start
        
        infrence_start = time.time()
        _, value, _ = agent.forward(observations, hidden_states)
        infrence_time += time.time() - infrence_start

        rollout_start = time.time()
        data = rollout.make_data(value, agent.std, agent.mean)
        rollout_time += time.time() - rollout_start

        normalizer_start = time.time()
        rewards_normalizer.add_variables(data.rewards)
        normalizer_time += time.time() - normalizer_start

        update_start = time.time()
        agent.updateModel(data.values, 
                          data.rewards, 
                          data.log_probs, 
                          data.advantage, 
                          data.entropy, 
                          rewards_normalizer.std(), 
                          rewards_normalizer.mean(), 
                          epoch)
        update_time += time.time() - update_start

        
        if should_render_agent and epoch % 10 == 0:
            render_queue.put(copy.deepcopy(agent.network))
        
        test_start = time.time()
        if epoch % test_epoch == 0:
            test_env = environment_function()
            test_observation = test_env.reset()
            test_rewards = 0
            test_hidden_state = torch.zeros((1,1, num_hidden_state))
            while True:
                policy, _, test_hidden_state = agent.forward(torch.tensor([test_observation], dtype=torch.float), test_hidden_state)
                action = agent.select_actions(policy)
                test_observation, ep_reward, done, _ = test_env.step(action[0].detach().numpy())
                test_rewards += ep_reward
                if done:
                    break
            writer.add_scalar("Reward", test_rewards, epoch)
            test_env.close()
        test_time += time.time() - test_start

        total_time = test_time + update_time + normalizer_time + rollout_time + envrironment_steps_time + infrence_time
            
if __name__ == "__main__":
    main()