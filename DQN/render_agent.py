import ray
import json
import gym
import numpy as np

from ray import tune
from ray.rllib.models import ModelCatalog
from DQN.dqn import DQNTrainer, DQNModel
from Env.predatorEnv import PredatorEnv


def env_creator(env_config):
    return PredatorEnv((20, 20), (17, 6), 100, (20, 20, 30, 10, 40), 20, 500)


if __name__ == "__main__":

    # Settings
    folder = "/home/tom/ray_results/DQNAlgorithm/DQNAlgorithm_predEnv_644d9_00000_0_2020-11-05_11-48-08"
    # env_name = "predEnv"
    checkpoint = 1873
    num_episodes = 1

    # Def env
    env = env_creator("")
    print(folder + "/params.json")

    ray.init()
    ModelCatalog.register_custom_model("DQNModel", DQNModel)

    # Load config
    with open(folder + "/params.json") as json_file:
        config = json.load(json_file)
    trainer = DQNTrainer(env=env, config=config)

    # Restore checkpoint
    trainer.restore(folder + "/checkpoint_{}/checkpoint-{}".format(checkpoint, checkpoint))

    avg_reward = 0
    for episode in range(num_episodes):
        step = 0
        total_reward = 0
        done = False
        observation = env.reset()

        while not done:
            step += 1
            env.render()
            print(observation)
            action, _, _ = trainer.get_policy().compute_actions([observation], [])
            observation, reward, done, info = env.step(action[0])
            total_reward += reward
        print("episode {} received reward {} after {} steps".format(episode, total_reward, step))
        avg_reward += total_reward
    print('avg reward after {} episodes {}'.format(avg_reward / num_episodes, num_episodes))
    env.close()
    del trainer
