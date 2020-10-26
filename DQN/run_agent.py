import ray
import os
from ray import tune
from ray.rllib.models import ModelCatalog

from dqn import DQNTrainer, DQNModel

if __name__ == "__main__":
    ray.init()
    ModelCatalog.register_custom_model("DQNModel", DQNModel)

    tune.run(
        DQNTrainer,
        # checkpoint_freq=10,
        checkpoint_at_end=True,
        stop={"episodes_total": 2000},
        config={
            "num_gpus": 0,
            "num_workers": 1,
            "framework": "torch",
            "rollout_fragment_length": 50,
            "env": "CartPole-v1",

            ########################################
            # Parameters Agent
            ########################################
            "lr": 0.004,
            # "lr": tune.grid_search([0.003, 0.004, 0.005, 0.006, 0.007]),
            "discount": 0.8,
            # "discount": tune.grid_search([0.5, 0.6, 0.7, 0.8, 0.9, 1]),
            "buffer_batch": 1500,
            # "buffer_batch": tune.grid_search([1000, 1500, 2500]),
            "epsilon": 0.6,
            # "epsilon": tune.grid_search([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]),
            "decay": 0.999,
            # "decay": tune.grid_search([0.99, 0.999, 0.9999]),
            "min_epsilon": 0.05,
            # "min_epsilon": tune.grid_search([0.01, 0.05, 0.1, 0.15, 0.2, 0.25]),

            "dqn_model": {
                "custom_model": "DQNModel",
                "custom_model_config": {
                },  # extra options to pass to your model
            }
        }
    )
