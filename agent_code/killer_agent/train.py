from collections import namedtuple, deque

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import random
from typing import List

import numpy as np
import torch
from torch import nn

import events as e
from .callbacks import state_to_features

from .architectures.networks import MyResNetBinary


"""
This agent is a very basic implementation of a reinforcement learning agent using Deep Q Learning.
It follows mostly the PyTorch tutorial given here: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
"""


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']


# hyperparameters
REPLAY_MEMORY_SIZE = 100000
BATCH_SIZE = 256
GAMMA = 0.9
EPS_START = 0.9
EPS_RB_START = 0.5
EPS_END = 0.05
EPS_DECAY = 5000
TAU = 0.001
INITIAL_LR = 1e-3
ITERATIONS_PER_ROUND = 20
LONG_UPDATE_EVERY_X = 100
ITERATIONS_EVERY_X = 500
UPDATE_AFTER_CYCLE = 1000

# This is only an example!
Transition = namedtuple('Transition',
                        ('img_state', 'binary_state', 'action', 'next_img_state', 'next_binary_state', 'reward'))

MOVE_AWAY = "MOVE_AWAY"
MOVE_TOWARDS = "MOVE_TOWARDS"
REVERSE_EVENT = "REVERSE"


def setup_training(self):
    self.model_dir.mkdir(exist_ok=True)
    self.model_dir_every_x = self.model_dir / "every_x"
    self.model_dir_every_x.mkdir(exist_ok=True)
    self.eps = EPS_START
    self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=INITIAL_LR, amsgrad=True)
    self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=5000, gamma=0.9)
    self.target_model = MyResNetBinary(in_channels=6, num_actions=len(ACTIONS), depth=3, num_base_channels=32, num_max_channels=512,
                                       blocks_per_layer=3, num_binary=1)
    self.target_model.load_state_dict(self.model.state_dict())
    self.target_model.to(self.device)
    self.criterion = nn.SmoothL1Loss()
    self.eps_mode = "cycle"
    self.eps_rb = EPS_RB_START
    self.previous_action = None
    self.model.train()
    self.target_model.eval()
    if (self.model_dir/"model_final.pth").is_file() and (self.model_dir/"target_model_final.pth").is_file() and \
        Path("logs/training_state.pickle").is_file():
        self.logger.info("loading model")
        self.model.load_state_dict(torch.load(self.model_dir/"model_final.pth"))
        self.target_model.load_state_dict(torch.load(self.model_dir/"target_model_final.pth"))
        with open("logs/training_state.pickle", "rb") as f:
            training_state = pickle.load(f)
        self.steps_done = training_state["steps_done"]
        self.epsilons = training_state["eps"]
        self.loss_list = training_state["loss"]
        self.total_rewards = training_state["total_rewards"]
        self.rewards_per_step = training_state["rewards_per_step"]
        self.ema_total_rewards = training_state["ema_total_rewards"]
        self.ema_rewards_per_step = training_state["ema_rewards_per_step"]
        with open("logs/deque.pickle", "rb") as f:
            self.replay_memory = pickle.load(f)
        #set scheduler to correct step
        for _ in range(self.steps_done):
            self.scheduler.step()
    else:
        self.steps_done = 0
        self.epsilons = []
        self.loss_list = []
        self.total_rewards = [0]
        self.rewards_per_step = []
        self.ema_total_rewards = []
        self.ema_rewards_per_step = []
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    old_img_features, old_binary_features = state_to_features(old_game_state)
    new_img_features, new_binary_features = state_to_features(new_game_state)
    old_field = old_img_features[0]
    old_agents = old_img_features[4]
    new_field = new_img_features[0]
    new_agents = new_img_features[4]
    old_agent_dist = old_field[old_agents == 1]
    # we don't want to punish our agent if the other agent moved
    new_agent_dist = new_field[old_agents == 1]
    old_nearest_agent = torch.tensor([-1.]) if torch.sum(old_agent_dist>0)==0 else torch.min(old_agent_dist[old_agent_dist>0])
    new_nearest_agent = torch.tensor([-1.]) if torch.sum(new_agent_dist>0)==0 else torch.min(new_agent_dist[new_agent_dist>0])
    # reward only when old and new nearest coin is one step apart
    if torch.isclose(torch.abs(old_nearest_agent-new_nearest_agent), torch.tensor([1/17])):
        if new_nearest_agent < old_nearest_agent:
            events.append(MOVE_TOWARDS)
            pass
        elif new_nearest_agent > old_nearest_agent:
            events.append(MOVE_AWAY)
            pass
    action = ACTIONS.index(self_action)
    if self.previous_action is not None:
        if (self_action == ACTIONS[0] and self.previous_action == ACTIONS[2]) or \
                (self_action == ACTIONS[1] and self.previous_action == ACTIONS[3]) or \
                (self_action == ACTIONS[2] and self.previous_action == ACTIONS[0]) or \
                (self_action == ACTIONS[3] and self.previous_action == ACTIONS[1]):
            events.append(REVERSE_EVENT)
    reward = reward_from_events(self, events)
    self.replay_memory.append(Transition(old_img_features, old_binary_features, torch.tensor([action]), \
                                        new_img_features, new_binary_features, torch.tensor([reward])))
    self.total_rewards[-1] += reward
    self.previous_action = self_action


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    reward = reward_from_events(self, events)
    last_img_features, last_binary_features = state_to_features(last_game_state)
    action = ACTIONS.index(last_action)
    self.replay_memory.append(Transition(last_img_features, last_binary_features, torch.tensor([action]), \
                                        None, None, torch.tensor([reward])))
    self.total_rewards[-1] += reward
    self.rewards_per_step.append(self.total_rewards[-1] / last_game_state['step'])
    if len(self.total_rewards) == 1:
        self.ema_total_rewards.append(self.total_rewards[-1])
    else:
        self.ema_total_rewards.append(0.99 * self.ema_total_rewards[-1] + 0.01 * self.total_rewards[-1])
    if len(self.rewards_per_step) == 1:
        self.ema_rewards_per_step.append(self.rewards_per_step[-1])
    else:
        self.ema_rewards_per_step.append(0.99 * self.ema_rewards_per_step[-1] + 0.01 * self.rewards_per_step[-1])
    optimize_model(self)
    update_eps(self)
    plot_and_save(self)
    self.previous_action = None
    self.total_rewards.append(0)


def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.CRATE_DESTROYED: 10,
        e.SURVIVED_ROUND: 50,
        e.KILLED_OPPONENT: 400,
        e.KILLED_SELF: -100,
        e.GOT_KILLED: -50,
        e.INVALID_ACTION: -100,
        e.WAITED: -10,
        MOVE_AWAY: -40,
        MOVE_TOWARDS: 20,
        REVERSE_EVENT: -10,
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum


def optimize_model(self):
    if len(self.replay_memory) < ITERATIONS_PER_ROUND*BATCH_SIZE:
        return
    loss_this_round = 0
    if (self.steps_done + 1) % EPS_DECAY == 0:
        iterations_this_round = UPDATE_AFTER_CYCLE
    elif (self.steps_done + 1) % LONG_UPDATE_EVERY_X == 0:
        iterations_this_round = ITERATIONS_EVERY_X
    else:
        iterations_this_round = ITERATIONS_PER_ROUND
    for iteration in range(iterations_this_round):
        transitions = random.sample(self.replay_memory, BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_img_state)), device=self.device, dtype=torch.bool)
        non_final_next_img_states = torch.stack([s for s in batch.next_img_state if s is not None]).to(dtype=torch.float32, device=self.device)
        non_final_next_binary_states = torch.stack([s for s in batch.next_binary_state if s is not None]).to(dtype=torch.float32, device=self.device)

        img_state_batch = torch.stack(batch.img_state).to(dtype=torch.float32, device=self.device)
        binary_state_batch = torch.stack(batch.binary_state).to(dtype=torch.float32, device=self.device)
        action_batch = torch.stack(batch.action).to(device=self.device)
        reward_batch = torch.stack(batch.reward).to(device=self.device)

        state_action_values = self.model(img_state_batch, binary_state_batch).gather(1, action_batch)
        
        next_state_values = torch.zeros(BATCH_SIZE, 1, device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_model(non_final_next_img_states, non_final_next_binary_states).max(1, keepdim=True)[0]
        
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        loss = self.criterion(state_action_values, expected_state_action_values)
        loss_this_round += loss.item()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 100)
        self.optimizer.step()
        if (iteration + 1) % ITERATIONS_PER_ROUND == 0:
            target_state_dict = self.target_model.state_dict()
            policy_state_dict = self.model.state_dict()
            for key in policy_state_dict:
                target_state_dict[key] = TAU * policy_state_dict[key] + (1 - TAU) * target_state_dict[key]
            self.target_model.load_state_dict(target_state_dict)
    self.scheduler.step()
    self.loss_list.append(loss_this_round / iterations_this_round)


def update_eps(self):
    """
    mode can either be 'decay' or 'cycle'
    'decay' decreases epsilon exponentially over EPS_DECAY steps
    'cycle' cycles epsilon with period EPS_DECAY. after first cycle, eps is only increased up to EPS_START_CYCLE
    """
    self.epsilons.append(self.eps)
    self.steps_done += 1
    # wait for 100 rounds and set eps_rb to False
    if self.eps_mode == "decay":
        self.eps = EPS_END + (EPS_START - EPS_END) * np.exp(-1. * self.steps_done / EPS_DECAY)
        self.eps_rb = self.eps
    elif self.eps_mode == "cycle":
        # cycling decay of eps over EPS_DECAY steps
        steps_cycle = self.steps_done % EPS_DECAY
        eps_max = EPS_START * 0.9 ** (self.steps_done // EPS_DECAY)
        self.eps = eps_max * np.exp(-2. * steps_cycle / EPS_DECAY)
        self.eps_rb = EPS_RB_START * 0.9 ** (self.steps_done // EPS_DECAY)
    else:
        raise NotImplementedError(f"mode {self.eps_mode} not implemented")


def plot_and_save(self):
    if (self.steps_done + 1) % 20 == 0:
        plt.figure(figsize=(8,15))
        plt.subplot(411)
        plt.title("total rewards")
        plt.plot(self.total_rewards, label="reward")
        plt.plot(self.ema_total_rewards, label="ema reward")
        plt.plot([0]*len(self.total_rewards), color="black", linestyle="--")
        plt.legend()
        plt.subplot(412)
        plt.title("rewards per step")
        plt.plot(self.rewards_per_step, label="reward")
        plt.plot(self.ema_rewards_per_step, label="ema reward")
        plt.plot([0]*len(self.rewards_per_step), color="black", linestyle="--")
        plt.legend()
        plt.subplot(413)
        plt.title("loss")
        plt.plot(self.loss_list)
        plt.yscale("log")
        plt.subplot(414)
        plt.plot(self.epsilons)
        plt.title("epsilon")
        plt.savefig("logs/rewards.png")
        plt.close()
    if (self.steps_done + 1) % 100 == 0:
        torch.save(self.model.state_dict(), self.model_dir_every_x/f"model_{self.steps_done + 1}.pth")
        torch.save(self.model.state_dict(), self.model_dir/"model_final.pth")
        torch.save(self.target_model.state_dict(), self.model_dir/"target_model_final.pth")
        training_state = {"steps_done": self.steps_done, "eps": self.epsilons, "loss": self.loss_list, \
                          "total_rewards": self.total_rewards, "rewards_per_step": self.rewards_per_step, \
                          "ema_total_rewards": self.ema_total_rewards, "ema_rewards_per_step": self.ema_rewards_per_step}
        with open("logs/training_state.pickle", "wb") as f:
            pickle.dump(training_state, f)
        with open("logs/deque.pickle", "wb") as f:
            pickle.dump(self.replay_memory, f)