from collections import deque

import numpy as np
from pathlib import Path
import torch

from .architectures.networks import *


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']


def setup(self):
    self.eps = 0
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.model_dir = Path("models")
    self.model = DQN(25, 6)
    if not self.train:
        self.model.load_state_dict(torch.load(self.model_dir/"model_final.pth", map_location=self.device))
    self.model.to(self.device)
    self.model.eval()


def act(self, game_state: dict) -> str:
    if self.train and np.random.random() < self.eps:
        self.logger.debug("Choosing action purely at random.")
        return np.random.choice(ACTIONS)

    with torch.no_grad():
        self.logger.debug("Querying model for action.")
        features = state_to_features(game_state)
        features = features.unsqueeze(0).to(dtype=torch.float32, device=self.device)
        q_values = self.model(features)
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
    return dist


def state_to_features(game_state: dict) -> np.array:
    field = game_state['field']
    bombs = game_state['bombs']
    coins = game_state['coins']
    explosion_map = game_state['explosion_map']
    agent = game_state['self']
    others = game_state['others']
    agent_x, agent_y = agent[3]
    agent_bomb = agent[2]
    field_features = find_shortest_paths((agent_x, agent_y), field)

    # calculate deadly for x steps
    deadly_for = np.zeros_like(field)
    for bomb in bombs:
        bomb_x, bomb_y = bomb[0]
        bomb_timer = bomb[1]+2
        for x in range(4):
            current_x = bomb_x - x
            if current_x < 0 or field[current_x, bomb_y] == -1:
                break
            if deadly_for[current_x, bomb_y] == 0 or bomb_timer < deadly_for[current_x, bomb_y]:
                deadly_for[current_x, bomb_y] = bomb_timer
        for x in range(4):
            current_x = bomb_x + x
            if current_x >= field.shape[0] or field[current_x, bomb_y] == -1:
                break
            if deadly_for[current_x, bomb_y] == 0 or bomb_timer < deadly_for[current_x, bomb_y]:
                deadly_for[current_x, bomb_y] = bomb_timer
        for y in range(4):
            current_y = bomb_y - y
            if current_y < 0 or field[bomb_x, current_y] == -1:
                break
            if deadly_for[bomb_x, current_y] == 0 or bomb_timer < deadly_for[bomb_x, current_y]:
                deadly_for[bomb_x, current_y] = bomb_timer
        for y in range(4):
            current_y = bomb_y + y
            if current_y >= field.shape[1] or field[bomb_x, current_y] == -1:
                break
            if deadly_for[bomb_x, current_y] == 0 or bomb_timer < deadly_for[bomb_x, current_y]:
                deadly_for[bomb_x, current_y] = bomb_timer

    deadly_for[(explosion_map==1)&(deadly_for==0)] = 1

    safe_to_go = np.where((deadly_for==0)&(field_features>=0), 1, -1)
    for x, y in zip(*np.where(deadly_for>2)):
        if field_features[x, y] < 0:
            continue
        way_to_safety = find_shortest_paths((x, y), field)
        if np.any((deadly_for==0)&(field_features>=0)&(way_to_safety!=-1)):
            min_safety_dist = np.min(way_to_safety[(deadly_for==0)&(field_features>=0)&(way_to_safety!=-1)])
            if min_safety_dist < deadly_for[x, y]-2:
                safe_to_go[x, y] = 1

    # (x, y), (x, y-1), (x+1, y), (x, y+1), (x-1, y)
    safe_steps = np.array([safe_to_go[agent_x, agent_y], safe_to_go[agent_x, agent_y-1], safe_to_go[agent_x+1, agent_y], \
                           safe_to_go[agent_x, agent_y+1], safe_to_go[agent_x-1, agent_y]])

    # (x, y-1), (x+1, y), (x, y+1), (x-1, y)
    next_coins = np.full(4, -1)
    for coin in coins:
        coin_path = find_shortest_paths(coin, field)
        if coin_path[agent_x, agent_y-1] != -1 and (next_coins[0] == -1 or coin_path[agent_x, agent_y-1] < next_coins[0]):
            next_coins[0] = coin_path[agent_x, agent_y-1]
        if coin_path[agent_x+1, agent_y] != -1 and (next_coins[1] == -1 or coin_path[agent_x+1, agent_y] < next_coins[1]):
            next_coins[1] = coin_path[agent_x+1, agent_y]
        if coin_path[agent_x, agent_y+1] != -1 and (next_coins[2] == -1 or coin_path[agent_x, agent_y+1] < next_coins[2]):
            next_coins[2] = coin_path[agent_x, agent_y+1]
        if coin_path[agent_x-1, agent_y] != -1 and (next_coins[3] == -1 or coin_path[agent_x-1, agent_y] < next_coins[3]):
            next_coins[3] = coin_path[agent_x-1, agent_y]

    # (x, y-1), (x+1, y), (x, y+1), (x-1, y)
    next_agents = np.full(4, -1)
    for other in others:
        other_path = find_shortest_paths(other[3], field)
        if other_path[agent_x, agent_y-1] != -1 and (next_agents[0] == -1 or other_path[agent_x, agent_y-1] < next_agents[0]):
            next_agents[0] = other_path[agent_x, agent_y-1]
        if other_path[agent_x+1, agent_y] != -1 and (next_agents[1] == -1 or other_path[agent_x+1, agent_y] < next_agents[1]):
            next_agents[1] = other_path[agent_x+1, agent_y]
        if other_path[agent_x, agent_y+1] != -1 and (next_agents[2] == -1 or other_path[agent_x, agent_y+1] < next_agents[2]):
            next_agents[2] = other_path[agent_x, agent_y+1]
        if other_path[agent_x-1, agent_y] != -1 and (next_agents[3] == -1 or other_path[agent_x-1, agent_y] < next_agents[3]):
            next_agents[3] = other_path[agent_x-1, agent_y]

    # how many crates can be blown up by placing a bomb at this position
    crates_to_blowup = np.zeros_like(field)
    crates = np.where(field == 1)
    for crate_x, crate_y in zip(*crates):
        for x in range(1, 4):
            current_x = crate_x - x
            if current_x < 0 or field[current_x, crate_y] == -1:
                break
            crates_to_blowup[current_x, crate_y] += 1
        for x in range(1, 4):
            current_x = crate_x + x
            if current_x >= field.shape[0] or field[current_x, crate_y] == -1:
                break
            crates_to_blowup[current_x, crate_y] += 1
        for y in range(1, 4):
            current_y = crate_y - y
            if current_y < 0 or field[crate_x, current_y] == -1:
                break
            crates_to_blowup[crate_x, current_y] += 1
        for y in range(1, 4):
            current_y = crate_y + y
            if current_y >= field.shape[1] or field[crate_x, current_y] == -1:
                break
            crates_to_blowup[crate_x, current_y] += 1
    crates_to_blowup[field_features<0] = 0

    # (x, y), (x, y-1), (x+1, y), (x, y+1), (x-1, y)
    crates_nearby = np.zeros(5)
    crates_nearby[0] = crates_to_blowup[agent_x, agent_y]
    crates_nearby[1] = crates_to_blowup[agent_x, agent_y-1]
    crates_nearby[2] = crates_to_blowup[agent_x+1, agent_y]
    crates_nearby[3] = crates_to_blowup[agent_x, agent_y+1]
    crates_nearby[4] = crates_to_blowup[agent_x-1, agent_y]
    
    for x, y in zip(*np.where(crates_to_blowup>0)):
        crate_path = find_shortest_paths((x, y), field)
        if crate_path[agent_x, agent_y-1] != -1 and crates_to_blowup[x, y] - crate_path[agent_x, agent_y-1] > crates_nearby[1]:
            crates_nearby[1] = crates_to_blowup[x, y] - crate_path[agent_x, agent_y-1]
        if crate_path[agent_x+1, agent_y] != -1 and crates_to_blowup[x, y] - crate_path[agent_x+1, agent_y] > crates_nearby[2]:
            crates_nearby[2] = crates_to_blowup[x, y] - crate_path[agent_x+1, agent_y]
        if crate_path[agent_x, agent_y+1] != -1 and crates_to_blowup[x, y] - crate_path[agent_x, agent_y+1] > crates_nearby[3]:
            crates_nearby[3] = crates_to_blowup[x, y] - crate_path[agent_x, agent_y+1]
        if crate_path[agent_x-1, agent_y] != -1 and crates_to_blowup[x, y] - crate_path[agent_x-1, agent_y] > crates_nearby[4]:
            crates_nearby[4] = crates_to_blowup[x, y] - crate_path[agent_x-1, agent_y]

    # (x, y-1), (x+1, y), (x, y+1), (x-1, y)
    valid_steps = np.array([field_features[agent_x, agent_y-1], field_features[agent_x+1, agent_y], \
                            field_features[agent_x, agent_y+1], field_features[agent_x-1, agent_y]])
    valid_steps = np.where(valid_steps>=0, 1, -1)

    # is it safe to drop a bomb
    safe_to_drop = False
    for x in range(1, 5):
        current_x = agent_x - x
        if current_x < 0 or field_features[current_x, agent_y] == -1:
            break
        if x == 4 and field_features[current_x, agent_y] >= 0:
            safe_to_drop = True
            break
        elif x < 4 and (field_features[current_x, agent_y-1] >= 0 or field_features[current_x, agent_y+1] >= 0):
            safe_to_drop = True
            break
    for x in range(1, 5):
        current_x = agent_x + x
        if current_x >= field_features.shape[0] or field_features[current_x, agent_y] == -1:
            break
        if x == 4 and field_features[current_x, agent_y] >= 0:
            safe_to_drop = True
            break
        elif x < 4 and (field_features[current_x, agent_y-1] >= 0 or field_features[current_x, agent_y+1] >= 0):
            safe_to_drop = True
            break
    for y in range(1, 5):
        current_y = agent_y - y
        if current_y < 0 or field_features[agent_x, current_y] == -1:
            break
        if y == 4 and field_features[agent_x, current_y] >= 0:
            safe_to_drop = True
            break
        elif y < 4 and (field_features[agent_x-1, current_y] >= 0 or field_features[agent_x+1, current_y] >= 0):
            safe_to_drop = True
            break
    for y in range(1, 5):
        current_y = agent_y + y
        if current_y >= field_features.shape[1] or field_features[agent_x, current_y] == -1:
            break
        if y == 4 and field_features[agent_x, current_y] >= 0:
            safe_to_drop = True
            break
        elif y < 4 and (field_features[agent_x-1, current_y] >= 0 or field_features[agent_x+1, current_y] >= 0):
            safe_to_drop = True
            break

    # 5 + 4 + 4 + 5 + 4 + 1 + 1 + 1 = 25
    features = np.concatenate([safe_steps, next_coins, next_agents, crates_nearby, valid_steps, \
                               [agent_x, agent_y, agent_bomb&safe_to_drop]])
    return torch.tensor(features).to(dtype=torch.float32)
