import gym
import numpy as np
import math

from gym import spaces
from world import World


class PredatorEnv(gym.Env):
    def __init__(self, map_size, prey_settings, prey_amount, pred_settings, pred_amount, max_t):
        self.world_settings = map_size, prey_settings, prey_amount, pred_settings, pred_amount, max_t
        self.world = World(map_size, prey_settings, prey_amount, pred_settings, pred_amount, max_t)

        self.action_space = spaces.Discrete(5)

        self.observation_space = spaces.Box(np.array([0, 0, -map_size[0], -map_size[1]]),
                                            np.array([pred_settings[0], math.inf, map_size[0], map_size[1]]))

    def reset(self):
        self.world = World(self.world_settings[0], self.world_settings[1], self.world_settings[3],
                           self.world_settings[4], self.world_settings[5], self.world_settings[6])

    def step(self, action):
        self.world.step(actions=action, env_type="pred")
        out = {
            "obs": self.world.get_pred_obs(),
            "rewards": self.world.get_pred_rewards(),
            "dones": self.world.get_pred_dones()
        }

        return out
