import json
import time

import numpy as np
import ray
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env

from DQN.dqn import DQNTrainer, DQNModel
from Env.predatorEnv import PredatorEnv


def env_creator(env_config):
    return PredatorEnv((20, 20), (17, 6), 100, (20, 20, 30, 10, 40), 20, 500)


if __name__ == "__main__":
    # Settings
    folder = "/home/tom/ray_results/DQNAlgorithm_2020-12-19_16-33-09/DQNAlgorithm_predEnv_7f7ae_00000_0_2020-12-19_16" \
             "-33-09"
    # env_name = "predEnv"
    checkpoint = 1852
    num_episodes = 2

    env = env_creator("")
    print(folder + "/params.json")

    ray.init()
    register_env("predEnv", env_creator)
    ModelCatalog.register_custom_model("DQNModel", DQNModel)

    # Load config
    with open(folder + "/params.json") as json_file:
        config = json.load(json_file)
    trainer = DQNTrainer(env="predEnv", config=config)

    # Restore checkpoint
    trainer.restore(folder + "/checkpoint_{}/checkpoint-{}".format(checkpoint, checkpoint))

    avg_reward = 0
    for episode in range(num_episodes):
        step = 0
        total_reward = 0
        done = False
        observation = env.reset()

        while not done:
            time.sleep(0.1)
            step += 1
            env.render()

            obs_batch = []
            for obs in observation.values():
                obs = np.array(obs)
                obs_batch.append(obs)

            action, _, _ = trainer.get_policy().compute_actions(obs_batch, [])
            action_dict = {}
            keys = list(observation.keys())

            for i, action in enumerate(action):
                action_dict[keys[i]] = action

            observation, rewards, done, info = env.step(action_dict)
            done = done.get("__all__")

            total_reward += sum(rewards.values())
        print("episode {} received reward {} after {} steps".format(episode, total_reward, step))
        avg_reward += total_reward
        env.world.stats.show_hist()
    print('avg reward after {} episodes {}'.format(avg_reward / num_episodes, num_episodes))
    env.close()
    del trainer
