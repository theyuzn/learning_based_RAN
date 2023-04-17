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
from torch.nn import functional as F
from torchvision import transforms as T

from ..algorithm.net import Decide_Grouping, LSTMDQN, ReplayMemory
from .action_space import grouping_action_space
from ..utils.constants import *
from ..envs.env import MAX_GROUP, state
from ..envs.ue import UE


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
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        np.random.seed(args.seed)

        # Log
        self.log_file = "grouping"
        self.logger = logging.getLogger('Grouping')
        self.logger.setLevel(logging.DEBUG)
        self.formatter = logging.Formatter('%(message)s')
        file_handler = logging.FileHandler(f'dqn_{self.log_file}_{args.model}.log')
        file_handler.setFormatter(self.formatter)
        self.logger.addHandler(file_handler)

        # Action apace
        self.action_space = grouping_action_space(action = -1, n = MAX_GROUP)

        # DQN Model
        self.dqn_hidden_state = self.dqn_cell_state = None
        self.target_hidden_state = self.target_cell_state = None
        self.mode: str = args.model.lower()
        if self.mode == 'dqn':
            self.net: Decide_Grouping = Decide_Grouping(self.action_space.get_dimension())
        elif self.mode == 'lstm':
            self.net: LSTMDQN = LSTMDQN(self.action_space.get_dimension())

            # For Optimization
            self.dqn_hidden_state, self.dqn_cell_state = self.net.init_states()
            self.target_hidden_state, self.target_cell_state = self.net.init_states()

            # For Training Play
            self.train_hidden_state, self.train_cell_state = self.net.init_states()

            # For Validation Play
            self.test_hidden_state, self.test_cell_state = self.net.init_states()

        if cuda:
            self.net.cuda()

        self.target: Decide_Grouping = copy.deepcopy(self.net)
        self.optimizer = optim.Adam(self.net.parameters(), lr=LEARNING_RATE)
        self.replay = ReplayMemory()
        self.epsilon = EPSILON_START


    def preprocessing(self, state : state):
        STEP = 0.2
        nrofULUE = len(state.ul_uelist)
        state_ndarray = np.empty(12)
        data_size_ndarray = np.empty(nrofULUE)
        delay_bound_ndarray = np.empty(nrofULUE)
        ue : UE
        i, state_idx, cnter = 0
        step = 0.
        total_size = 0

        ### Initial all the data
        for ue in state.ul_uelist :
            data_size_ndarray[i] = ue.sizeOfData
            total_size += ue.sizeOfData
            delay_bound_ndarray[i] = ue.delay_bound
            i += 1
        
        data_size_ndarray.sort(axis=None, kind='quicksort')
        delay_bound_ndarray.sort(axis=None, kind='quicksort')
        size_range = data_size_ndarray[nrofULUE] - data_size_ndarray[0]
        delay_range = delay_bound_ndarray[nrofULUE] - delay_bound_ndarray[0]

        state_ndarray[state_idx] = nrofULUE
        state_idx += 1
        state_ndarray[state_idx] = total_size
        state_idx += 1

        ### Calculate the number of UE in each step of size of data
        i = 0
        step = 0.
        for i  in range(nrofULUE) :
            if data_size_ndarray[i]/size_range <= step and data_size_ndarray[i]/size_range > step :
                cnter += 1
            else:
                state_ndarray[state_idx] = cnter
                state_idx += 1
                cnter = 0
                step += STEP


        ### Calculate the number of UE in each step of delay bound
        i = 0
        step = 0.
        for i  in range(nrofULUE) :
            if delay_bound_ndarray[i]/delay_range <= step and delay_bound_ndarray[i]/delay_range > step :
                cnter += 1
            else:
                state_ndarray[state_idx] = cnter
                state_idx += 1
                cnter = 0
                step += STEP


        ### Return the processed state array
        return state_ndarray



    def select_action(self, states: state) -> tuple:
        # Decrease epsilon value
        self.epsilon = EPSILON_END + (EPSILON_START - EPSILON_END) * math.exp(-1. * self.step / EPSILON_DECAY)

        # Randomly select a grouping number
        if self.epsilon > random():
            sample_action = random.randrange(MAX_GROUP) + 1
            action = torch.LongTensor([[sample_action]])
            return action
        
        state_array : np.ndarray
        state_array = self.preprocessing(state = states)
        state_array = torch.as_tensor(state_array)

        if self.mode == 'dqn':
            action = self.net(state_array).data.cpu().max(1)[1]
        elif self.mode == 'lstm':
            action, self.dqn_hidden_state, self.dqn_cell_state = self.net(state_array, self.train_hidden_state, self.train_cell_state)
            action = action.data.cpu().max(1)[1]

        return action
