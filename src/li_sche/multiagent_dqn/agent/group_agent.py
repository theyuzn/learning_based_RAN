import argparse
import copy
import logging
import math

from collections import deque
from collections import namedtuple
# from random import random, sample
import random

import numpy as np
import pylab
import torch
from PIL import Image
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import transforms as T

from ..algorithm.algorithm import DQN, LSTMDQN, ReplayMemory
from .action_space import grouping_action_space
from ..utils.constants import *
from ..envs.env import MAX_GROUP, state


class GroupAgent:
    def __init__(self, args: argparse.Namespace, cuda = True, action_repeat: int = 4):
        self.args = args
        self.clip: bool = args.clip
        self.seed: int = args.seed
        self.action_repeat: int = action_repeat
        self.frame_skipping: int = args.skip_action
        self._state_buffer = deque(maxlen=self.action_repeat)
        self.step = 0
        self.best_score = args.best or -10000
        self.best_count = 0
        self._play_steps = deque(maxlen=5)

        # torch init
        # Random Seed
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        np.random.seed(args.seed)

        # Log
        self.logger = logging.getLogger('DQN')
        self.logger.setLevel(logging.DEBUG)
        self.formatter = logging.Formatter('%(message)s')

        file_handler = logging.FileHandler(f'dqn_{args.model}.log')
        file_handler.setFormatter(self.formatter)
        self.logger.addHandler(file_handler)

        # Action apace
        self.action_space = grouping_action_space(action = -1, n = MAX_GROUP)


        # DQN Model
        self.dqn_hidden_state = self.dqn_cell_state = None
        self.target_hidden_state = self.target_cell_state = None

        self.mode: str = args.model.lower()
        if self.mode == 'dqn':
            self.dqn: DQN = DQN(self.action_space.get_dimension())
        elif self.mode == 'lstm':
            self.dqn: LSTMDQN = LSTMDQN(self.action_space.get_dimension())

            # For Optimization
            self.dqn_hidden_state, self.dqn_cell_state = self.dqn.init_states()
            self.target_hidden_state, self.target_cell_state = self.dqn.init_states()

            # For Training Play
            self.train_hidden_state, self.train_cell_state = self.dqn.init_states()

            # For Validation Play
            self.test_hidden_state, self.test_cell_state = self.dqn.init_states()

        if cuda:
            self.dqn.cuda()

        # DQN Target Model
        self.target: DQN = copy.deepcopy(self.dqn)

        # Optimizer
        self.optimizer = optim.Adam(self.dqn.parameters(), lr=LEARNING_RATE)

        # Replay Memory
        self.replay = ReplayMemory()

        # Epsilon
        self.epsilon = EPSILON_START




    def select_action(self, states: state) -> tuple:
        # Decrease epsilon value
        self.epsilon = EPSILON_END + (EPSILON_START - EPSILON_END) * math.exp(-1. * self.step / EPSILON_DECAY)

        # Randomly select a grouping number
        if self.epsilon > random():
            sample_action = random.randrange(MAX_GROUP) + 1
            action = torch.LongTensor([[sample_action]])
            return action

        states = states.reshape(1, self.action_repeat, self.env.width, self.env.height)
        states_variable: Variable = Variable(torch.FloatTensor(states).cuda())

        if self.mode == 'dqn':
            states_variable.volatile = True
            action = self.dqn(states_variable).data.cpu().max(1)[1]
        elif self.mode == 'lstm':
            action, self.dqn_hidden_state, self.dqn_cell_state = self.dqn(states_variable, self.train_hidden_state, self.train_cell_state)
            action = action.data.cpu().max(1)[1]

        return action
