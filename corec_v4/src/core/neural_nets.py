import torch
import torch.nn as nn
import torch.optim as optim
import time

class NanoNeuralNet(nn.Module):
    def __init__(self, input_size=6, hidden_size=8, output_size=3):
        super(NanoNeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self._memoria = []

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        self._memoria.append((x, time.time()))
        if len(self._memoria) > 50:
            self._memoria.pop(0)
        return x

class DeepQNetwork(nn.Module):
    def __init__(self, state_size=4, action_size=3):
        super(DeepQNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 4)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(4, action_size)
        self._memoria = []

    def forward(self, state):
        x = self.fc1(state)
        x = self.relu(x)
        x = self.fc2(x)
        self._memoria.append((state, x))
        if len(self._memoria) > 100:
            self._memoria.pop(0)
        return x