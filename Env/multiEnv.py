import gym
import numpy as np
from gym import spaces
from ray.rllib.env import MultiAgentEnv

from Env.world import World


class MultiEnv(gym.Env, MultiAgentEnv):
    def __init__(self, map_size, prey_settings, prey_amount, pred_settings, pred_amount, max_t):
        self.world_settings = map_size, prey_settings, prey_amount, pred_settings, pred_amount, max_t
        self.world = World(map_size, prey_settings, prey_amount, pred_settings, pred_amount, max_t)

        self.action_space_pred = spaces.Discrete(5)
        self.action_space_prey = spaces.Discrete(4)

        self.observation_space_pred = spaces.Box(low=np.array([0, 0, -map_size[0], -map_size[1]]),
                                                 high=np.array([pred_settings[0], np.inf, map_size[0], map_size[1]]))
        self.observation_space_prey = spaces.Box(low=np.array([0, -map_size[0], -map_size[1]]),
                                                 high=np.array([prey_settings[0], map_size[0], map_size[1]]))

    def reset(self):
        self.world = World(self.world_settings[0], self.world_settings[1], self.world_settings[2],
                           self.world_settings[3], self.world_settings[4], self.world_settings[5])
        return self.world.get_obs()

    def step(self, action):
        self.world.step(actions=action, env_type="multi")
        return self.world.get_obs(), self.world.get_rewards(), self.world.get_dones(), {}

    def render(self, **kwargs):
        self.world.render()
