import gym
import numpy as np

from gym import spaces
from world import World


class PreyEnv(gym.Env):
    def __init__(self, map_size, prey_settings, prey_amount, pred_settings, pred_amount, max_t):
        self.world_settings = map_size, prey_settings, prey_amount, pred_settings, pred_amount, max_t
        self.world = World(map_size, prey_settings, prey_amount, pred_settings, pred_amount, max_t)

        self.action_space = spaces.Discrete(4)

        self.observation_space = spaces.Box(np.array([0, -map_size[0], -map_size[1]]),
                                            np.array([prey_settings[0], map_size[0], map_size[1]]))

    def reset(self):
        self.world = World(self.world_settings[0], self.world_settings[1], self.world_settings[3],
                           self.world_settings[4], self.world_settings[5], self.world_settings[6])

    def step(self, actions):
        self.world.step(actions=actions, env_type="prey")
        out = {
            "obs": self.world.get_prey_obs(),
            "rewards": self.world.get_prey_rewards(),
            "dones": self.world.get_prey_dones()
        }

        return out
