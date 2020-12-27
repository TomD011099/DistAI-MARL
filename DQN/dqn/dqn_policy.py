import math
from collections import deque

import random
import torch
import numpy as np
from statistics import mean
from ray.rllib.policy import Policy
from ray.rllib.models import ModelCatalog
from torch.nn import MSELoss
import torch.nn.functional as F
import torch.nn as nn


class DQNPolicy(Policy):
    """Hand-coded policy that returns random actions."""

    def __init__(self, observation_space, action_space, config):
        Policy.__init__(self, observation_space, action_space, config)
        self.observation_space = observation_space
        self.action_space = action_space
        self.config = config

        self.lr = self.config["lr"]  # Extra options need to be added in dqn.py


        # GPU settings
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        self.dtype_f = torch.FloatTensor
        self.dtype_l = torch.LongTensor
        self.dtype_b = torch.BoolTensor
        if self.use_cuda:
            print("Using CUDA")
            self.dtype_f = torch.cuda.FloatTensor
            self.dtype_l = torch.cuda.LongTensor
            self.dtype_b = torch.cuda.BoolTensor

        self.dqn_model = ModelCatalog.get_model_v2(
            obs_space=self.observation_space,
            action_space=self.action_space,
            num_outputs=4,
            name="DQNModel",
            model_config=self.config["dqn_model"],
            framework="torch",
        ).to(self.device, non_blocking=True)
        # self.replay_buffer = deque(maxlen=self.config["bufferlen"])
        # self.buffer_batch = self.config["buffer_batch"]
        self.discount = torch.tensor(self.config["discount"]).to(self.device, non_blocking=True)
        self.optim = torch.optim.Adam(self.dqn_model.parameters(), lr=self.lr)
        self.MSE_loss_fn = MSELoss(reduction='mean')
        self.epsilon = self.config["epsilon"]
        self.min_epsilon = self.config["min_epsilon"]
        self.decay = self.config["decay"]

    def compute_actions(self,
                        obs_batch,
                        state_batches=None,
                        prev_action_batch=None,
                        prev_reward_batch=None,
                        info_batch=None,
                        episodes=None,
                        explore=None,
                        timestep=None,
                        **kwargs):
        # Worker function

        obs_batch_t = torch.tensor(obs_batch).type(torch.FloatTensor)
        value = self.dqn_model(obs_batch_t)
        actions = torch.argmax(value, 1)

        epsilon_log = []
        for index in range(len(actions)):
            self.epsilon = max(self.epsilon * self.decay, self.min_epsilon)
            epsilon_log.append(self.epsilon)
            if np.random.random() < self.epsilon:
                actions[index] = self.action_space.sample()

        actions = actions.cpu().detach().tolist()

        return actions, [], {"epsilon_log": epsilon_log}
        # return [self.action_space.sample() for _ in obs_batch], [], {}

    def learn_on_batch(self, samples):
        # Trainer function

        # Sample has following fields:
        #
        #   t                 incrementing number that sometimes rolls back to 0
        #   eps_id            unique identifier for an episode
        #   agent_index       uniquely identifies an agent within an episode
        # Outputs from interacting with the environment:
        #   obs
        #   actions
        #   rewards
        #   prev_actions
        #   prev_rewards
        #   dones
        #   infos
        #   new_obs
        # Uniquely identifies a sample batch. This is important to distinguish RNN
        # sequences from the same episode when multiple sample batches are
        # concatenated (fusing sequences across batches can be unsafe).
        #   unroll_id

        epsilon_log = samples["epsilon_log"]

        # Oplossingscode voor memory
        """obs = samples["obs"]
        new_obs = samples["new_obs"]
        rewards = samples["rewards"]
        actions = samples["actions"]
        dones = samples["dones"]
        batch = zip(obs, new_obs, rewards, actions, dones)
        for obs_s, new_obs_s, rewards_s, actions_s, dones_s in batch:
            self.replay_buffer.append([obs_s, new_obs_s, rewards_s, actions_s, dones_s])"""

        # Mijn code voor memory
        """for i in range(len(samples["dones"])):
            state = samples["obs"][i]
            action = samples["actions"][i]
            reward = samples["rewards"][i]
            done = samples["dones"][i]
            new_state = samples["new_obs"][i]
            self.replay_buffer.append((state, action, reward, done, new_state))"""

        """if len(self.replay_buffer) < self.buffer_batch:
            return {"learner_stats": {"loss": 0, "epsilon": mean(epsilon_log), "buffer_size": len(self.replay_buffer)}}

        batch = random.sample(self.replay_buffer, self.buffer_batch)
        samples = {"obs": [None for _ in range(self.buffer_batch)], "new_obs": [None for _ in range(self.buffer_batch)],
                   "rewards": [None for _ in range(self.buffer_batch)],
                   "actions": [None for _ in range(self.buffer_batch)],
                   "dones": [None for _ in range(self.buffer_batch)]}

        i = 0
        for sample in batch:
            samples["obs"][i] = sample[0]
            samples["new_obs"][i] = sample[1]
            samples["rewards"][i] = sample[2]
            samples["actions"][i] = sample[3]
            samples["dones"][i] = sample[4]
            i += 1"""

        """batch = random.sample(list(self.replay_buffer), sample_size)
        batch = list(zip(*batch))
        obs_batch_t = torch.cat((torch.tensor(np.array(samples["obs"])).type(torch.FloatTensor),
                                 torch.tensor(np.array(batch[0])).type(torch.FloatTensor)))
        rewards_batch_t = torch.cat((torch.tensor(np.array(samples["rewards"])).type(torch.FloatTensor),
                                     torch.tensor(np.array(batch[2])).type(torch.FloatTensor)))
        new_obs_batch_t = torch.cat((torch.tensor(np.array(samples["new_obs"])).type(torch.FloatTensor),
                                     torch.tensor(np.array(batch[4])).type(torch.FloatTensor)))
        actions_batch_t = torch.cat((torch.tensor(np.array(samples["actions"])).type(torch.LongTensor),
                                     torch.tensor(np.array(batch[1])).type(torch.LongTensor)))
        dones_t = torch.cat((torch.tensor(np.array(samples["dones"])).type(torch.BoolTensor),
                             torch.tensor(np.array(batch[3])).type(torch.BoolTensor)))"""

        obs_batch_t = torch.tensor(np.array(samples["obs"])).to(self.device, non_blocking=True).type(self.dtype_f)
        new_obs_batch_t = torch.tensor(np.array(samples["new_obs"])).to(self.device, non_blocking=True).type(
            self.dtype_f)
        rewards_batch_t = torch.tensor(np.array(samples["rewards"])).to(self.device, non_blocking=True).type(
            self.dtype_f)
        actions_batch_t = torch.tensor(np.array(samples["actions"])).to(self.device, non_blocking=True).type(
            self.dtype_l)
        dones_t = torch.tensor(np.array(samples["dones"])).to(self.device, non_blocking=True).type(self.dtype_b)

        # actions_batch_t = actions_batch_t.unsqueeze(-1)
        guess = self.dqn_model(obs_batch_t).gather(1, actions_batch_t.unsqueeze(-1)).squeeze(-1)
        max_q = self.dqn_model(new_obs_batch_t).max(1)[0].detach()
        max_q[dones_t] = 0.0
        target = rewards_batch_t + (self.discount * max_q)
        target = target.detach()
        loss = self.MSE_loss_fn(guess, target)
        self.optim.zero_grad()
        loss.backward()
        for param in self.dqn_model.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optim.step()
        return {"learner_stats": {"loss": loss.cpu().item(), "epsilon": mean(epsilon_log)}}

    def get_weights(self):
        # Trainer function
        weights = {}
        weights["dqn_model"] = self.dqn_model.cpu().state_dict()
        self.dqn_model.to(self.device, non_blocking=False)
        return weights

    def set_weights(self, weights):
        # Worker function
        if "dqn_model" in weights:
            self.dqn_model.load_state_dict(weights["dqn_model"], strict=True)
            self.dqn_model.to(self.device, non_blocking=False)
