from env_wrapper import SubprocVecEnv, DummyVecEnv
import numpy as np
import multiagent.scenarios as scenarios
from multiagent.environment import MultiAgentEnv

def make_parallel_env(n_rollout_threads, seed=1):
    def get_env_fn(rank):
        def init_env():
            env = make_env("simple_adversary")
            env.seed(seed + rank * 1000)
            np.random.seed(seed + rank * 1000)
            return env
        return init_env
#    if n_rollout_threads == 1:
#        return DummyVecEnv([get_env_fn(0)])
#    else:
    return SubprocVecEnv([get_env_fn(i) for i in range(n_rollout_threads)])


def make_env(scenario_name,benchmark=False):

    scenario = scenarios.load(scenario_name + ".py").Scenario()
    world = scenario.make_world()
    env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)

    return env