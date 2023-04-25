import argparse
import copy
import glob
import logging
import math
import os
import re
import sys

from collections import deque
from collections import namedtuple
from random import random, randrange, sample

import numpy as np
import torch
from PIL import Image
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import transforms as T


from envs.env import Env, MAX_GROUP, State
from li_sche.multiagent_dqn.agent import Agent, Joint_Action
from net.brain import ReplayMemory


class Brain():
    ### Init
    def __init__(self, args: argparse.Namespace, cuda = True, action_repeat: int = 4):
        self.action_repeat = action_repeat
        self.memory = ReplayMemory(capacity = 50000)

        # Environment
        self.env = Env(args=args)
        self.step = 0

        # Agent
        self.agent = Agent(args = args, cuda=True, action_repeat=self.action_repeat)

 
    ### Get initial states
    def get_initial_states(self):
        state = self.env.reset()
        return state
    
    ########################## Replay buffer ##########################
    def store_state(self, state):
        self.memory.push(state)

    def recent_states(self):
        return self._state_buffer
    ###################################################################



    ############################### Test ############################## 
    def test_system(self):
        state = self.env.reset()
        slot = 0
        done = False
        while not done:
            ul_uelist = state.ul_uelist

            used_rb = 0
            if state.schedule_slot_info == 'U':
                for i in range(len(ul_uelist)):
                    ul_uelist[i].set_Group(randrange(4))
                    ul_uelist[i].set_RB(1)
                    used_rb += 1
                    # print(rf'used_rb ${used_rb}')
                    if used_rb > 246:
                        break

            state, reward, done = self.env.step(action = ul_uelist)
            slot += 1
            print(f'{state}, {reward}, {done}', end = '\n')
    ###################################################################



    def DL_slot(self):
        return


    def Special_slot(self):
        return
    


    ############################ Training #############################
    def train(self, gamma: float = 0.99):
        # Initial States
        reward_sum = 0.
        q_mean = [0., 0.]
        target_mean = [0., 0.]

        while True:
            ### states is an np.stack which stores preprocessed state -- np.ndarray
            # states : np.ndarray = self.get_initial_states()
            state : State =  self.env.reset()
            cumulated_reward = 0
            losses = []
            target_update_flag = False
            play_flag = False
            play_steps = 0
            real_play_count = 0
            real_score = 0
            done = False

            while True:        
                ## state is used to check which the schedule slot is. Ex: 'D', 'U', 'S'
                ## preprocessed_state is processed state, type : np.ndarray
                schudule_slot_info = state.get_schedule_slot_info()
                joint_action : Joint_Action = Joint_Action()

                match schudule_slot_info:
                    case 'D':
                        # skip
                        joint_action = self.DL_slot()
                    case 'S':
                        # skip
                        joint_action = self.Special_slot()
                    case 'U':
                        joint_action = self.agent.select_action(state=state)
                
                next_state, reward, done = self.env.step(joint_action)
                
                if done:
                    self.memory.push(state, joint_action, reward, None)
                else:
                    self.memory.push(state, joint_action, reward, next_state)
                
                reward_sum += reward
           
                # Change States
                state = next_state
                

                # Optimize
                # if self.replay.is_available():
                    # loss, reward_sum, q_mean, target_mean = self.optimize(gamma = gamma)
                    # losses.append(loss[0])

                if done:
                    break

                # Increase step
                self.step += 1
                play_steps += 1             
            
            break      
    ###################################################################



    ######################### Optmization #############################
    def optimize(self, gamma: float):

        if len(self.replay) < BATCH_SIZE:
            return

        # Get Samples : return Transition(*zip(transitions))
        batch = self.replay.sample(BATCH_SIZE)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(BATCH_SIZE, device=self.device)

       

        return 
    ###################################################################


    def save_checkpoint(self, filename='dqn_checkpoints/checkpoint.pth.tar'):
        dirpath = os.path.dirname(filename)

        if not os.path.exists(dirpath):
            os.mkdir(dirpath)

        checkpoint = {
            'dqn': self.dqn.state_dict(),
            'target': self.target.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'step': self.step,
            'best': self.best_score,
            'best_count': self.best_count
        }
        torch.save(checkpoint, filename)

    def load_checkpoint(self, filename='dqn_checkpoints/checkpoint.pth.tar', epsilon=None):
        checkpoint = torch.load(filename)
        self.dqn.load_state_dict(checkpoint['dqn'])
        self.target.load_state_dict(checkpoint['target'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.step = checkpoint['step']
        self.best_score = self.best_score or checkpoint['best']
        self.best_count = checkpoint['best_count']

    def load_latest_checkpoint(self, epsilon=None):
        r = re.compile('chkpoint_(dqn|lstm)_(?P<number>-?\d+)\.pth\.tar$')

        files = glob.glob(f'dqn_checkpoints/chkpoint_{self.mode}_*.pth.tar')

        if files:
            files = list(map(lambda x: [int(r.search(x).group('number')), x], files))
            files = sorted(files, key=lambda x: x[0])
            latest_file = files[-1][1]
            self.load_checkpoint(latest_file, epsilon=epsilon)
            print(f'latest checkpoint has been loaded - {latest_file}')
        else:
            print('no latest checkpoint')


    # @property
    # def play_step(self):
    #     return np.nan_to_num(np.mean(self._play_steps))

    # def _sum_params(self, model):
    #     return np.sum([torch.sum(p).data[0] for p in model.parameters()])

    # def imshow(self, sample_image: np.array, transpose=False):
    #     if transpose:
    #         sample_image = sample_image.transpose((1, 2, 0))
    #     pylab.imshow(sample_image, cmap='gray')
    #     pylab.show()

