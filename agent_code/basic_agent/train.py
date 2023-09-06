from collections import namedtuple, deque

import torch
from typing import List

import events as e
from .callbacks import state_to_features

from .architectures.networks import MyResNetBinary


"""
This agent is a very basic implementation of a reinforcement learning agent using Deep Q Learning.
It follows mostly the PyTorch tutorial given here: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
"""


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']


# hyperparameters
REPLAY_MEMORY_SIZE = 10000
BATCH_SIZE = 64
GAMMA = 0.9
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TAU = 0.001
INITIAL_LR = 1e-4

# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 3  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...

# Events
PLACEHOLDER_EVENT = "PLACEHOLDER"


def setup_training(self):
    self.replay_memory = deque(maxlen=10000)
    if not self.continue_training:
        self.steps_done = 0
        self.eps = EPS_START
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=INITIAL_LR, amsgrad=True)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=0.9)
    else:
        self.steps_done = self.steps_done
        self.eps = EPS_START
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=INITIAL_LR, amsgrad=True)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=0.9)
        self.model.load_state_dict(torch.load("my-saved-model.pt"))
    self.target_model = MyResNetBinary(1, len(ACTIONS), depth=3, num_base_channels=8, num_max_channels=64, blocks_per_layer=2)
    self.target_model.load_state_dict(self.model.state_dict())


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    pass


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    pass


def reward_from_events(self, events: List[str]) -> int:
    pass
