import numpy as np
import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from collections import deque
from collections import namedtuple
from random import random, sample

# Training
BATCH_SIZE = 32

# Replay Memory
REPLAY_MEMORY = 50000

# Epsilon
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 100000

# LSTM Memory
LSTM_MEMORY = 128

# ETC Options
TARGET_UPDATE_INTERVAL = 1000
CHECKPOINT_INTERVAL = 5000
PLAY_INTERVAL = 900
PLAY_REPEAT = 1
LEARNING_RATE = 0.0001

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


### Torch example
# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         # 1 input image channel, 6 output channels, 5x5 square convolution
#         # kernel
#         self.conv1 = nn.Conv2d(1, 6, 5)
#         self.conv2 = nn.Conv2d(6, 16, 5)
#         # an affine operation: y = Wx + b
#         self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 5*5 from image dimension
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, 10)

#     def forward(self, x):
#         # Max pooling over a (2, 2) window
#         x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
#         # If the size is a square, you can specify with a single number
#         x = F.max_pool2d(F.relu(self.conv2(x)), 2)
#         x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x

class DQN(nn.Module):
    def __init__(self, n_action):
        super(DQN, self).__init__()
        self.n_action = n_action

        self.linear1 = nn.Linear(12,24)
        self.linear2 = nn.Linear(24,24)
        self.linear3 = nn.Linear(24,8)
        self.linear4 = nn.Linear(8, self.n_action)
        self.softmax = nn.Softmax()


    def forward(self, state):
        x = self.linear1(state_variable)
        x = self.linear2(x)
        x = self.linear3(x)
        x = self.linear4(x)

        y = F.softmax(x, dim = 0)
        
        return y
    
class UE_schedule(nn.Module):
    def __init__(self, n_action):
        super(UE_schedule, self).__init__()
        self.n_action = n_action

        self.encoder = nn.Sequential(
            nn.Linear(12, 8),
            nn.Sigmoid(),
            nn.Linear(8, 8),
            nn.Sigmoid(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

        self.decoder = nn.Sequential(
            nn.Linear(13, 24),
            nn.Linear(24, 64),
            nn.Linear(64, 64),
            nn.Linear(64, self.n_action),
        )
    
    def set_n(self, n):
        self.n_action = n

    def forward(self, state_variable):
        ## To decide perform contention or scheduling ##
        x = self.encoder(state_variable)

        ## Processor
        x = x.detach().numpy()[0]
        if x < 0.5:
            x = 0
        else:
            x = 1
        state_variable = state_variable.detach().numpy()
        state_variable = np.append(state_variable, x)
        state_variable = torch.as_tensor(state_variable, dtype = torch.float)

        ## To decide which group index ##
        y = self.decoder(state_variable) # Tensor
        y = F.softmax(y, dim = 0)
        return x, y