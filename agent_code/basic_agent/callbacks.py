from collections import deque

import numpy as np
from pathlib import Path
import torch

from .architectures.networks import MyResNetBinary
from .callbacks_rule_based import act_rule_based, setup_rule_based


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']


def setup(self):
    self.eps = 0
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.model_dir = Path("models")
    self.model = MyResNetBinary(in_channels=5, num_actions=len(ACTIONS), depth=3, num_base_channels=32, num_max_channels=512,
                                blocks_per_layer=2, num_binary=1)
    if not self.train:
        self.model.load_state_dict(torch.load(self.model_dir/"model_final.pth"))
    self.model.to(self.device)
    self.model.eval()
    setup_rule_based(self)


def act(self, game_state: dict) -> str:
    if self.train and np.random.random() < self.eps:
        if np.random.random() < self.eps_rb:
            self.logger.debug("Acting according to rule based agent.")
            return act_rule_based(self, game_state)
        self.logger.debug("Choosing action purely at random.")
        return np.random.choice(ACTIONS)

    with torch.no_grad():
        self.logger.debug("Querying model for action.")
        img_features, binary_features = state_to_features(game_state)
        img_features = img_features.unsqueeze(0).to(dtype=torch.float32, device=self.device)
        binary_features = binary_features.unsqueeze(0).to(dtype=torch.float32, device=self.device)
        q_values = self.model(img_features, binary_features)
        action = torch.argmax(q_values).item()
        return ACTIONS[action]


def find_shortest_paths(agent_pos: tuple[int, int], field: np.ndarray) -> np.ndarray:
    """
    Finds the shortest paths from the agent to all other points on the field.
    Agent can move only horizontally and vertically and only on free tiles
    distance is measured as the number of steps needed to reach the point, unreachable points are given as -1
    implemented as a breadth-first search
    :param agent_pos: The position of the agent, given as (x, y).
    :param field: The field, given as a numpy array. Crates are given as 1, walls as -1 and free tiles as 0.
    :return: A numpy array of the same size as the field, where each entry contains the length of the shortest path
    """
    rows, cols = field.shape
    dist = np.full(field.shape, -1, dtype=float)
    dist[agent_pos] = 0

    queue = deque([agent_pos])

    moves = [(0, 1), (1, 0), (0, -1), (-1, 0)]

    while queue:
        current = queue.popleft()
        x, y = current
        for dx, dy in moves:
            nx, ny = x + dx, y + dy
            if 0 <= nx < rows and 0 <= ny < cols and field[nx, ny] == 0 and dist[nx, ny] == -1:
                dist[nx][ny] = dist[x][y] + 1
                queue.append((nx, ny))
    # normalize positive distances
    dist[dist > 0] /= rows
    return dist


def state_to_features(game_state: dict) -> np.array:
    field = game_state['field']
    bombs = game_state['bombs']
    coins = game_state['coins']
    explosion_map = game_state['explosion_map']
    agent = game_state['self']
    agent_pos = agent[3]
    agent_bomb = agent[2]
    others = game_state['others']
    others_pos = [other[3] for other in others]
    others_bombs = [other[2] for other in others]
    field_features = find_shortest_paths(agent_pos, field)
    bombs_features = np.zeros_like(field_features)
    # 0 is no danger, 4 is max danger
    for bomb in bombs:
        bombs_features[bomb[0]] = 4-bomb[1]
    coin_features = np.zeros_like(field_features)
    agent_features = np.zeros_like(field_features)
    for coin in coins:
        coin_features[coin] = 1
    for other_pos in others_pos:
        agent_features[other_pos] = 1
    features = np.stack([field_features, bombs_features, explosion_map, coin_features, agent_features])
    return torch.tensor(features), torch.tensor([agent_bomb])
