import os
import pickle
import random

import numpy as np


"""
This agent is a very basic implementation of a reinforcement learning agent using Deep Q Learning.
It follows mostly the PyTorch tutorial given here: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
"""


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']


def setup(self):
    pass


def act(self, game_state: dict) -> str:
    pass


def state_to_features(game_state: dict) -> np.array:
    pass
