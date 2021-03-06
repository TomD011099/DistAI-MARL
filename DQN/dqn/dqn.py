from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

from ray.rllib.agents import with_common_config
from ray.rllib.agents.trainer_template import build_trainer

from DQN.dqn.dqn_policy import DQNPolicy

logger = logging.getLogger(__name__)

DEFAULT_CONFIG = with_common_config({
    ########################################
    # Parameters Agent
    ########################################
    "lr": 0,
    "bufferlen": 10000,
    "buffer_batch": 500,
    "discount": 0.9,
    "epsilon": 1,
    "decay": 0.9995,
    "min_epsilon": 0.05,

    "dqn_model": {
        "custom_model": "DQNModel",
        "custom_model_config": {
            "network_size": [32,64,32]
        },  # extra options to pass to your model
    }
})

DQNTrainer = build_trainer(
    name="DQNAlgorithm",
    default_policy=DQNPolicy,
    default_config=DEFAULT_CONFIG)
