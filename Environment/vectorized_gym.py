import cloudpickle
import gym
import multiprocessing as mp
import numpy as np
import torch

from typing import Any, Callable

class CloudpickleWrapper:
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    :param var: the variable you wish to wrap for pickling with cloudpickle
    """

    def __init__(self, var: Any):
        self.var = var

    def __getstate__(self) -> Any:
        return cloudpickle.dumps(self.var)

    def __setstate__(self, var: Any) -> None:
        self.var = cloudpickle.loads(var)

class Result:
    def __init__(self):
        self.observation = []
        self.reward = []
        self.info = []
        self.done = []

    def add_agent_result(self, observation : Any, reward : float, info : dict, done : bool):
        self.observation.append(observation)
        self.reward.append(reward)
        self.info.append(info)
        self.done.append(done)

    def merge_other(self, other):
        self.observation += other.observation
        self.reward += other.reward
        self.info += other.info
        self.done += other.done


def _worker(
    remote,
    parent_remote,
    env_fn_wrapper: CloudpickleWrapper,
    num_environments: int,
) -> None: 
    assert num_environments > 0 
    environments = [env_fn_wrapper.var() for _ in range(num_environments)]
    parent_remote.close()
    while True:
        try:
            cmd, data = remote.recv()
            if cmd == "step":
                results = Result()
                for action, env in zip(data, environments):
                    # print("EEHEHEHEH", action)
                    observation, reward, done, _ = env.step(action.detach().numpy())
                    if done:
                        observation = env.reset()
                    results.add_agent_result(observation, reward, {}, done)
                remote.send(results)
            elif cmd == "reset":
                observations = []
                for env in environments:
                    observations.append(env.reset())
                remote.send(observations)
            elif cmd == "close":
                for env in environments:
                    env.close()
                remote.close()
                break
            elif cmd == "get_spaces":
                remote.send((environments[0].observation_space, environments[0].action_space))
            else:
                raise NotImplementedError(f"`{cmd}` is not implemented in the worker")
        except EOFError:
            break

class GymOnSubprocesses:
    """ 
    `gym_function` should be a function the returns the required gym.
    `num_processes` is how many processes should be started. Must be 
                    divisible by `num_groups`.
    `agents_per_process` is how many agents to start per process.
    `num_groups` is how many groups to split the processes into. .
    """ 
    def __init__(
        self, 
        gym_fucntion: Callable[[], gym.Env], 
        num_processes : int, 
        agents_per_process : int):

        assert num_processes > 0

        self.num_processes = num_processes
        self.agents_per_process = agents_per_process

        self.waiting = False
        forkserver_available = "forkserver" in mp.get_all_start_methods()
        start_method = "forkserver" if forkserver_available else "spawn"
        ctx = mp.get_context(start_method)

        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(num_processes)])
        self.processes = []
        for work_remote, remote in zip(self.work_remotes, self.remotes):
            args = (work_remote, remote, CloudpickleWrapper(gym_fucntion), agents_per_process)
            process = ctx.Process(target=_worker, args=args, daemon=True)
            process.start()
            self.processes.append(process)
            work_remote.close()

        self.remotes[0].send(("get_spaces", None))
        self.observation_space, self.action_space = self.remotes[0].recv()

    def step_async(self, actions: np.ndarray) -> None:
        assert len(actions) == self.num_processes * self.agents_per_process
        assert not self.waiting
        actions = np.split(actions, self.num_processes)
        for remote, process_actions in zip(self.remotes, actions):
            assert len(process_actions) == self.agents_per_process
            remote.send(("step", process_actions))
        self.waiting = True

    def step_wait(self):
        assert self.waiting
        results = Result()
        for remote in self.remotes:
            results.merge_other(remote.recv())

        self.waiting= False
        return results

    def reset(self):
        observations = []
        for remote in self.remotes:
            remote.send(("reset", None))
            observations += remote.recv()
        return torch.tensor(np.stack(observations))

    def close(self):
        for remote in self.remotes:
            remote.send(("close", None))
        
