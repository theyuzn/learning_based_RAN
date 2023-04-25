import argparse
import copy
import math

from collections import deque
from collections import namedtuple
from random import random, randrange

import numpy as np
import pylab
import torch
from PIL import Image
from torch import nn, optim
from torch.nn import functional as F
from torchvision import transforms as T

from ..algorithm.net import UE_schedule
from .action_space import ue_action_space
from ..utils.constants import *
from ..envs.env import MAX_GROUP


class UEAgent:

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
        self.action_space = ue_action_space(action = -1, n = MAX_GROUP)

        # DQN Model
        self.net: UE_schedule = UE_schedule(self.action_space.get_dimension())
        # self.net.cuda()

        self.target: UE_schedule = copy.deepcopy(self.net)
        self.optimizer = optim.Adam(self.net.parameters(), lr=LEARNING_RATE)
        self.epsilon = EPSILON_START



    def dicision_action(self, state: np.ndarray, nrof_group : int) -> tuple:
        # Decrease epsilon value
        self.epsilon = EPSILON_END + (EPSILON_START - EPSILON_END) * math.exp(-1. * self.step / EPSILON_DECAY)

        # Randomly select a grouping number
        if self.epsilon > random():
            action = randrange(nrof_group)
            return action
        
        self.net.set_n(nrof_group)
        state_array = torch.as_tensor(state, dtype = torch.float)
        x, y = self.net.forward(state_variable = state_array)
        action_prob = y.detach().numpy()

        action = 0
        while action == 0:
            for i in range(len(action_prob)):
                if 1 - action_prob[i] < random():
                    action = i # group index
        action = action * x
        return action
    

