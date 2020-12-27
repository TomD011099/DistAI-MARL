import ray
import os
import gym
from ray import tune
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env

from DQN.dqn import DQNTrainer, DQNModel
from Env import predatorEnv
from Env.predatorEnv import PredatorEnv
from Env.preyEnv import PreyEnv


def env_creator(env_config):
    return PreyEnv((30, 30), (17, 6), 100, (20, 20, 30, 10, 40), 20, 200)
    # return PredatorEnv((30, 30), (17, 6), 100, (20, 20, 30, 10, 40), 20, 500)


if __name__ == "__main__":
    ray.init()
    register_env("preyEnv", env_creator)
    ModelCatalog.register_custom_model("DQNModel", DQNModel)

    tune.run(
        DQNTrainer,
        # checkpoint_freq=10,
        checkpoint_at_end=True,
        stop={"episodes_total": 10000},
        config={
            "num_gpus": 0,
            "num_workers": 1,
            "framework": "torch",
            "rollout_fragment_length": 50,
            "env": "preyEnv",

            ########################################
            # Parameters Agent
            ########################################
            "lr": 4e-3,
            # "lr": tune.grid_search([0.01, 0.015, 0.02, 0.025, 0.03, 0.035]),
            "discount": 0.985,
            # "discount": tune.grid_search([0.7, 0.75, 0.8, 0.85, 0.9, 0.95]),
            "bufferlen": 200000,
            "buffer_batch": 2000,
            # "buffer_batch": tune.grid_search([1000, 2000, 3000]),
            "epsilon": 1,
            # "epsilon": tune.grid_search([0.7, 0.8, 0.9, 1]),
            "decay": 0.99998,
            # "decay": tune.grid_search([0.999999, 0.9999999]),
            # "min_epsilon": 1,
            "min_epsilon": tune.grid_search([0.01, 1]),

            "dqn_model": {
                "custom_model": "DQNModel",
                "custom_model_config": {
                },  # extra options to pass to your model
            },
            "evaluation_interval": 100,
            "evaluation_num_episodes": 100,
            "evaluation_config": {
                "epsilon": -1
            }
        }
    )
