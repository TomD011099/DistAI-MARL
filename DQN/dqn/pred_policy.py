import random
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F
from ray.rllib.models import ModelCatalog
from ray.rllib.policy import Policy


class PredPolicy(Policy):
    """Hand-coded policy that returns random actions."""

    def __init__(self, observation_space, action_space, config):
        Policy.__init__(self, observation_space, action_space, config)
        self.observation_space = observation_space
        self.action_space = action_space
        self.config = config

        self.lr = self.config["lr"]  # Extra options need to be added in dqn.py
        self.discount = self.config["discount"]

        # GPU settings
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")

        self.dqn_model = ModelCatalog.get_model_v2(
            obs_space=self.observation_space,
            action_space=self.action_space,
            num_outputs=5,
            name="DQNModel",
            model_config=self.config["dqn_model"],
            framework="torch",
        ).to(self.device, non_blocking=True)
        self.replay_buffer = deque(maxlen=self.config["bufferlen"])
        self.buffer_batch = self.config["buffer_batch"]
        self.optim = torch.optim.Adam(self.dqn_model.parameters(), self.lr)
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

        self.epsilon = max(self.epsilon * self.decay, self.min_epsilon)
        if random.random() < self.epsilon:
            return [self.action_space.sample() for _ in obs_batch], [], {}
        else:
            obs_batch_t = torch.tensor(obs_batch).type(torch.FloatTensor)
            value = self.dqn_model(obs_batch_t)
            indices = torch.max(value, 1)[1]
            return indices.numpy(), [], {}

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

        for i in range(len(samples["dones"])):
            state = samples["obs"][i]
            action = samples["actions"][i]
            reward = samples["rewards"][i]
            done = samples["dones"][i]
            new_state = samples["new_obs"][i]
            self.replay_buffer.append((state, action, reward, done, new_state))

        if len(self.replay_buffer) < self.buffer_batch:
            sample_size = len(self.replay_buffer)
        else:
            sample_size = self.buffer_batch
        batch = random.sample(list(self.replay_buffer), sample_size)
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
                             torch.tensor(np.array(batch[3])).type(torch.BoolTensor)))

        actions_batch_t = actions_batch_t.unsqueeze(-1)
        guess = self.dqn_model(obs_batch_t).gather(1, actions_batch_t)
        max_q = self.dqn_model(new_obs_batch_t).detach().max(1)[0]
        max_q[dones_t] = 0
        target = rewards_batch_t + (self.discount * max_q)
        loss = F.mse_loss(guess, target.unsqueeze(1))
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        return {"learner_stats": {"loss": loss.item()}}

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
