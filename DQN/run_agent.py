import ray
from ray import tune
from ray.rllib.agents import with_common_config
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env

from DQN.dqn import DQNTrainer, DQNModel, DQNPolicy
from DQN.dqn.dqn import DEFAULT_CONFIG
from DQN.dqn.pred_policy import PredPolicy
from DQN.dqn.prey_policy import PreyPolicy
from Env.multiEnv import MultiEnv


def env_creator(config):
    return MultiEnv((20, 20), (17, 6), 100, (20, 20, 30, 10, 40), 20, 500)


def policy_mapping_fn(agent_id):
    if agent_id.startswith("pred"):
        return "pred"
    else:
        return "prey"


if __name__ == "__main__":
    pred_c = {"num_pred": 20, "num_prey": 100, "action_space": 5}
    prey_c = {"num_pred": 20, "num_prey": 100, "action_space": 4}

    ray.init()
    env = env_creator(DEFAULT_CONFIG)
    register_env("multiEnv", env_creator)
    ModelCatalog.register_custom_model("DQNModel", DQNModel)

    policies = {
        "pred": (DQNPolicy,
                 env.observation_space_pred,
                 env.action_space_pred,
                 DEFAULT_CONFIG),
        "prey": (DQNPolicy,
                 env.observation_space_prey,
                 env.action_space_prey,
                 DEFAULT_CONFIG)
    }

    tune.run(
        DQNTrainer,
        checkpoint_at_end=True,
        stop={"episodes_total": 2000},
        config={
            "num_gpus": 0,
            "num_workers": 1,
            "framework": "torch",
            "rollout_fragment_length": 50,

            ########################################
            # Parameters Agent
            ########################################
            "lr": 0.001,
            # "lr": tune.grid_search([0.001, 0.0025, 0.005, 0.01, 0.015]),
            "discount": 0.8,
            # "discount": tune.grid_search([0.5, 0.6, 0.7, 0.8, 0.9, 1]),
            "buffer_batch": 1500,
            # "buffer_batch": tune.grid_search([1000, 1500, 2500]),
            "epsilon": 0.8,
            # "epsilon": tune.grid_search([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]),
            "decay": 0.999999,
            # "decay": tune.grid_search([0.9, 0.99, 0.999, 0.9999, 0.99999, 0.999999]),
            "min_epsilon": 0.05,
            # "min_epsilon": tune.grid_search([0.01, 0.05, 0.1, 0.15, 0.2, 0.25]),

            "env": "multiEnv",
            "multiagent": {
                "policy_mapping_fn": policy_mapping_fn,
                "policies": policies,
                "policies_to_train": policies
            },
            "dqn_model": {
                "custom_model": "DQNModel",
                "custom_model_config": {
                    "network_size":[32,64,32]
                },  # extra options to pass to your model
            },
            "evaluation_interval": 100,  # based on training iterations
            "evaluation_num_episodes": 100,
            "evaluation_config": {
                "epsilon": -1,
            },
        }
    )
