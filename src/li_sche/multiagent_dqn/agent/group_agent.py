import argparse
import copy
import math

from collections import deque
from collections import namedtuple
# from random import random, sample
from random import random, randrange

import numpy as np
import torch
from PIL import Image
from torch import nn, optim
import torch.nn.functional as F

from ..algorithm.net import Decide_Grouping
from .action_space import grouping_action_space
from ..utils.constants import *
from ..envs.env import MAX_GROUP

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

        # Action apace
        self.action_space = grouping_action_space(action = -1, n = MAX_GROUP)

        # DQN Model
        self.net: Decide_Grouping = Decide_Grouping(self.action_space.get_dimension())
        # self.net.cuda()

        self.target: Decide_Grouping = copy.deepcopy(self.net)
        self.optimizer = optim.Adam(self.net.parameters(), lr=LEARNING_RATE)
        self.epsilon = EPSILON_START


    def select_action(self, state: np.ndarray) -> tuple:
        # Decrease epsilon value
        self.epsilon = EPSILON_END + (EPSILON_START - EPSILON_END) * math.exp(-1. * self.step / EPSILON_DECAY)

        # Randomly select a grouping number
        if self.epsilon > random():
            # 1 ~ 4
            action = randrange(MAX_GROUP) + 1 
            return action
        
        state_array = torch.as_tensor(state, dtype = torch.float)
        action_prob = self.net.forward(state_variable = state_array)
        action_prob = action_prob.detach().numpy()

        action = 0
        while action == 0:
            for i in range(len(action_prob)):
                if 1 - action_prob[i] < random():
                    action = i + 1
        return action