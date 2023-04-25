import random
from collections import namedtuple, deque
from .constant import REPLAY_MEMORY, BATCH_SIZE
import torch
import numpy as np
from random import sample


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