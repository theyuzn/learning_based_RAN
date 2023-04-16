import numpy as np
import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from collections import deque
from collections import namedtuple
from random import random, sample


from .constant import *

class ReplayMemory(object):
    def __init__(self, capacity=REPLAY_MEMORY):
        self.capacity = capacity
        self.memory = deque(maxlen=self.capacity)
        self.Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state'))
        self._available = False

    def put(self, state: np.array, action: torch.LongTensor, reward: np.array, next_state: np.array):
        state = torch.FloatTensor(state)
        reward = torch.FloatTensor([reward])
        if next_state is not None:
            next_state = torch.FloatTensor(next_state)
        transition = self.Transition(state=state, action=action, reward=reward, next_state=next_state)
        self.memory.append(transition)

    def sample(self, batch_size):
        transitions = sample(self.memory, batch_size)
        return self.Transition(*(zip(*transitions)))

    def size(self):
        return len(self.memory)

    def is_available(self):
        if self._available:
            return True

        if len(self.memory) > BATCH_SIZE:
            self._available = True
        return self._available

class DQN(nn.Module):
    def __init__(self, n_action):
        super(DQN, self).__init__()
        self.n_action = n_action

        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0)

        self.affine1 = nn.Linear(3136, 512)
        self.affine2 = nn.Linear(512, self.n_action)

    def forward(self, x):
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))

        # print(h.size())
        # print(h.view(h.size(0), -1).size())

        h = F.relu(self.affine1(h.view(h.size(0), -1)))
        h = self.affine2(h)
        return h


class LSTMDQN(nn.Module):
    def __init__(self, n_action):
        super(LSTMDQN, self).__init__()
        self.n_action = n_action

        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.lstm = nn.LSTM(16, LSTM_MEMORY, 1)  # (Input, Hidden, Num Layers)

        self.affine1 = nn.Linear(LSTM_MEMORY * 64, 512)
        # self.affine2 = nn.Linear(2048, 512)
        self.affine2 = nn.Linear(512, self.n_action)

    def forward(self, x, hidden_state, cell_state):
        # CNN
        h = F.relu(F.max_pool2d(self.conv1(x), kernel_size=2, stride=2))
        h = F.relu(F.max_pool2d(self.conv2(h), kernel_size=2, stride=2))
        h = F.relu(F.max_pool2d(self.conv3(h), kernel_size=2, stride=2))
        h = F.relu(F.max_pool2d(self.conv4(h), kernel_size=2, stride=2))

        # LSTM
        h = h.view(h.size(0), h.size(1), 16)  # (32, 64, 4, 4) -> (32, 64, 16)
        h, (next_hidden_state, next_cell_state) = self.lstm(h, (hidden_state, cell_state))
        h = h.view(h.size(0), -1)  # (32, 64, 256) -> (32, 16348)

        # Fully Connected Layers
        h = F.relu(self.affine1(h.view(h.size(0), -1)))
        # h = F.relu(self.affine2(h.view(h.size(0), -1)))
        h = self.affine2(h)
        return h, next_hidden_state, next_cell_state

    def init_states(self) -> [Variable, Variable]:
        hidden_state = Variable(torch.zeros(1, 64, LSTM_MEMORY).cuda())
        cell_state = Variable(torch.zeros(1, 64, LSTM_MEMORY).cuda())
        return hidden_state, cell_state

    def reset_states(self, hidden_state, cell_state):
        hidden_state[:, :, :] = 0
        cell_state[:, :, :] = 0
        return hidden_state.detach(), cell_state.detach()