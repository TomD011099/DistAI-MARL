import random
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F
from ray.rllib.models import ModelCatalog
from ray.rllib.policy import Policy
from torch.nn import MSELoss


class DQNPolicy(Policy):
    """Hand-coded policy that returns random actions."""

    def __init__(self, observation_space, action_space, config):
        Policy.__init__(self, observation_space, action_space, config)
        self.observation_space = observation_space
        self.action_space = action_space
        self.config = config
        self.action_shape = action_space.n

        # GPU settings
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        self.dtype_f = torch.FloatTensor
        self.dtype_l = torch.LongTensor
        self.dtype_b = torch.BoolTensor
        if self.use_cuda:
            self.dtype_f = torch.cuda.FloatTensor
            self.dtype_l = torch.cuda.LongTensor
            self.dtype_b = torch.cuda.BoolTensor

        self.lr = self.config["lr"]  # Extra options need to be added in dqn.py
        self.discount = torch.tensor(self.config["discount"]).to(self.device, non_blocking=True)
        self.buffer_batch = self.config["buffer_batch"]
        self.epsilon = self.config["epsilon"]
        self.min_epsilon = self.config["min_epsilon"]
        self.decay = self.config["decay"]

        self.replay_buffer = deque(maxlen=self.config["bufferlen"])

        self.dqn_model = ModelCatalog.get_model_v2(
            obs_space=self.observation_space,
            action_space=self.action_space,
            num_outputs=5,
            name="DQNModel",
            model_config=self.config["dqn_model"],
            framework="torch",
        ).to(self.device, non_blocking=True)
        self.MSE_loss_fn = MSELoss(reduction='mean')
        self.optimizer = torch.optim.Adam(self.dqn_model.parameters(), lr=self.lr)

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
        obs_batch_t = torch.tensor(obs_batch).type(self.dtype_f)
        q_value_batch_t = self.dqn_model(obs_batch_t)
        action_batch_t = torch.argmax(q_value_batch_t, axis=1)

        for index in range(len(action_batch_t)):
            self.epsilon *= self.decay
            if self.epsilon < self.min_epsilon:
                self.epsilon = self.min_epsilon
            if np.random.random() < self.epsilon:
                action_batch_t[index] = random.randint(0, self.action_shape - 1)

        action = action_batch_t.cpu().detach().tolist()
        return action, [], {}

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

        obs_batch_t = torch.tensor(np.array(samples["obs"])).to(self.device, non_blocking=True).type(self.dtype_f)
        rewards_batch_t = torch.tensor(np.array(samples["rewards"])).to(self.device, non_blocking=True).type(self.dtype_f)
        new_obs_batch_t = torch.tensor(np.array(samples["new_obs"])).to(self.device, non_blocking=True).type(self.dtype_f)
        actions_batch_t = torch.tensor(np.array(samples["actions"])).to(self.device, non_blocking=True).type(self.dtype_l)
        dones_t = torch.tensor(np.array(samples["dones"])).to(self.device, non_blocking=True).type(self.dtype_b)

        guess = self.dqn_model(obs_batch_t).gather(1, actions_batch_t.unsqueeze(-1)).squeeze(-1)
        max_q = self.dqn_model(new_obs_batch_t).detach().max(1)[0].detach()
        max_q[dones_t] = 0.0
        target = rewards_batch_t + (self.discount * max_q)
        target = target.detach()
        loss_t = self.MSE_loss_fn(guess, target)

        self.optimizer.zero_grad()
        loss_t.backward()
        self.optimizer.step()

        return {"learner_stats": {"loss": loss_t.cpu().item()}}

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
