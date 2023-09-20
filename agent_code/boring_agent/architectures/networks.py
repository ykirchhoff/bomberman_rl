import torch
from torch import nn


class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, output_size)
        self.nonlin = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        x = self.nonlin(self.fc1(x))
        x = self.nonlin(self.fc2(x))
        x = self.nonlin(self.fc3(x))
        return self.fc4(x)