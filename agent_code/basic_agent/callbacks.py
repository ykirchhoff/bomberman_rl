import os
import random

import numpy as np
import torch

from .architectures.networks import MyResNetBinary


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']


def setup(self):
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.model = MyResNetBinary(1, len(ACTIONS), depth=3, num_base_channels=8, num_max_channels=64, blocks_per_layer=2)
    if not self.train:
        self.model.load_state_dict(torch.load("my-saved-model.pt"))
    self.model.to(self.device)


def act(self, game_state: dict) -> str:
    if self.train and np.random.random() < self.eps:
        self.logger.debug("Choosing action purely at random.")
        return np.random.choice(ACTIONS)

    with torch.no_grad():
        self.logger.debug("Querying model for action.")
        state = torch.tensor(state_to_features(game_state)).float().unsqueeze(0).to(self.device)
        q_values = self.model(state)
        action = torch.argmax(q_values).item()
        return ACTIONS[action]


# suggested by copilot, not yet tested
def get_neighbors(pos, shape):
    """
    returns the neighbors of a position on the field
    :param pos: position on the field
    :param shape: shape of the field
    :return: list of neighbors
    """
    neighbors = []
    if pos[0] > 0:
        neighbors.append((pos[0] - 1, pos[1]))
    if pos[0] < shape[0] - 1:
        neighbors.append((pos[0] + 1, pos[1]))
    if pos[1] > 0:
        neighbors.append((pos[0], pos[1] - 1))
    if pos[1] < shape[1] - 1:
        neighbors.append((pos[0], pos[1] + 1))
    return neighbors


# suggested by copilot, not yet tested
def find_shortest_path(agent_pos, field):
    """
    calculates the shortest path from the position of the agent to the other tiles of the field
    the agent can only move horizontally or vertically and can't move through walls or crates (field value 1 or -1)
    we use a breadth-first search to find the shortest path
    :param agent_pos: current position of the agent
    :param field: field as given by the game state
    """
    field = np.where(field == 1, -1, field)  # replace all walls with -1
    field = np.where(field == 2, 1, field)  # replace all crates with 1
    field = np.where(field == 0, 2, field)  # replace all free tiles with 2
    field = np.where(field == -1, 0, field)  # replace all walls with 0
    field[agent_pos] = 3  # set position of agent to 3

    # initialize the distance matrix and the queue
    distance = np.zeros_like(field)
    queue = []
    queue.append(agent_pos)

    # loop through the queue until it is empty
    while len(queue) > 0:
        # get the first element of the queue
        pos = queue.pop(0)
        # get the distance of the current position
        dist = distance[pos]
        # get the possible neighbors of the current position
        neighbors = get_neighbors(pos, field.shape)
        # loop through the neighbors
        for n in neighbors:
            # if the neighbor is a free tile
            if field[n] == 2:
                # set the distance of the neighbor to the distance of the current position + 1
                distance[n] = dist + 1
                # set the neighbor to a crate
                field[n] = 1
                # add the neighbor to the queue
                queue.append(n)
            # if the neighbor is a coin
            elif field[n] == 5:
                # set the distance of the neighbor to the distance of the current position + 1
                distance[n] = dist + 1
                # set the neighbor to a crate
                field[n] = 1
                # add the neighbor to the queue
                queue.append(n)
                # return the distance matrix
                return distance
    # return the distance matrix
    return distance


def state_to_features(game_state: dict) -> np.array:
    field = game_state['field']
    bombs = game_state['bombs']
    coins = game_state['coins']
    explosion_map = game_state['explosion_map']
    agent = game_state['self']
    agent_pos = agent[3]
    agent_bomb = agent[2]
    others = [o[3] for o in game_state['others']]
    pass