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

class Shrared_RB_DQN(nn.Module):
    def __init__(self,n):
        super(Shrared_RB_DQN, self).__init__()
        self.output = n
        
        self.linear = nn.Sequential(
            nn.Linear(12, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, self.output)
        )

    def forward(self, state):
        x = self.linear(state)
        y = F.softmax(x, dim = 0)
        return y

class Shared_DQN(nn.Module):
    def __init__(self, n):
        super(Shared_DQN, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(12,128),
            nn.ReLU(),
            nn.Linear(128,128),
            nn.ReLU(),
            nn.Linear(128,4),
            nn.ReLU()
        ) 

    def forward(self, state):
        return self.linear(state)

class Resource_DQN(nn.Module):
    def __init__(self, n):
        super(Resource_DQN, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(12,128),
            nn.ReLU(),
            nn.Linear(128,32),
            nn.Linear(32,n)
        )

    def forward(self, state):
        x = self.linear(state)
        y = F.softmax(x, dim = 0)
        return y
    
class MCS_DQN(nn.Module):
    def __init__(self, n):
        super(Resource_DQN, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(13,128),
            nn.ReLU(),
            nn.Linear(128,32),
            nn.Linear(32,n)
        )

    def forward(self, state):
        x = self.linear(state)
        y = F.softmax(x, dim = 0)
        return y
    

class UE_Classification_DQN(nn.Module):
    def __init__(self, n):
        super(Resource_DQN, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(16,128),
            nn.ReLU(),
            nn.Linear(128,32),
            nn.Linear(32,n)
        )

    def forward(self, state):
        x = self.linear(state)
        y = F.softmax(x, dim = 0)
        return y