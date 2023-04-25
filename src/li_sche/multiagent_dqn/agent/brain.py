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
import pylab
import torch
from PIL import Image
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import transforms as T


from ..envs.env import Env, MAX_GROUP, State
from ..envs.ue import UE
from ..algorithm.net import *
from ..algorithm.replay_buffer import ReplayMemory
from ..algorithm.constant import *
from .group_agent import GroupAgent
from .rb_action import RBAction
from .ue_agent import UEAgent

class Brain():
    ### Init
    def __init__(self, args: argparse.Namespace, cuda = True, action_repeat: int = 4):
        self.action_repeat = action_repeat
        self.replay = ReplayMemory(capacity = 50000)
        self._state_buffer = deque(maxlen = self.action_repeat)
        # Environment
        self.env = Env(args=args)
        self.step = 0
        # Agents
        self.grouping_action = GroupAgent(args=args)
        self.ue_action = UEAgent(args=args)

 
    ### Get initial states
    def get_initial_states(self):
        state = self.env.reset()
        processed_state = self.preprocessing(state)
        states = np.stack([processed_state for _ in range(self.action_repeat)], axis=0)
        self._state_buffer = deque(maxlen = self.action_repeat)
        for _ in range(self.action_repeat):
            self._state_buffer.append(processed_state)
        return states
    
    ########################## Replay buffer ##########################
    def add_state(self, state):
        self._state_buffer.append(state)

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

    ########################## Preprocessing ##########################
    def preprocessing(self, state : State):
        if len(state.ul_uelist) < 1:
            return np.empty(0)
        
        STEP = 0.2 # For 20% step in range
        nrofULUE = len(state.ul_uelist)
        state_ndarray = np.zeros(12)
        data_size_ndarray = np.zeros(nrofULUE)
        delay_bound_ndarray = np.zeros(nrofULUE)
        ue : UE
        state_idx = 0
        total_size = 0

        ### Initial all the data
        i = 0
        for ue in state.ul_uelist :
            data_size_ndarray[i] = ue.sizeOfData
            total_size += ue.sizeOfData
            delay_bound_ndarray[i] = ue.delay_bound
            i += 1

        state_ndarray[state_idx] = nrofULUE
        state_idx += 1
        state_ndarray[state_idx] = total_size
        state_idx += 1
        
        data_size_ndarray.sort(axis=0, kind='mergesort')
        size_top = data_size_ndarray[nrofULUE - 1]
        size_bottom = data_size_ndarray[0]
        size_range = size_top - size_bottom

        step = 0.2
        for size in data_size_ndarray:
            while size > size_range*step + size_bottom:
                state_idx +=1
                step += STEP
            state_ndarray[state_idx] += 1
            
        state_idx += 1

        delay_bound_ndarray.sort(axis=0, kind='mergesort')
        delay_top = delay_bound_ndarray[nrofULUE - 1]
        delay_bottom = delay_bound_ndarray[0]
        delay_range = delay_top - delay_bottom

        step = 0.2
        for delay in delay_bound_ndarray:
            while delay > delay_range*step + delay_bottom:
                state_idx +=1
                step += STEP
            state_ndarray[state_idx] += 1
        state_idx += 1

        ### Return the processed state array
        return state_ndarray
    ###################################################################


    def DL_slot(self):
        return


    def Special_slot(self):
        return
    
    ############################## Train ##############################
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
                preprocessed_state : np.ndarray = self.preprocessing(state)

                nrof_group = 0          # The decision of number of group
                action_uelist = []      # The action list of UL UEs

                match schudule_slot_info:
                    case 'D':
                        # skip
                        self.DL_slot()
                    case 'S':
                        # skip
                        self.Special_slot()
                    case 'U':
                        if len(state.ul_uelist) > 0:
                            nrof_group = self.grouping_action.select_action(preprocessed_state)
                            ## All UE are scheduled
                            if nrof_group == 1:
                                for ue in state.ul_uelist:
                                    ue.set_Group(0)
                                    ue.set_RB(1)
                                    action_uelist.append(ue)
                            
                            ## Except group#0, all of other groups are set to contention 
                            else:
                                for ue in state.ul_uelist:
                                    group_index = self.ue_action.dicision_action(preprocessed_state, nrof_group)
                                    ue.set_RB(1)
                                    action_uelist.append(ue)
                
                next_state, reward, done = self.env.step(action_uelist)
                self.add_state(self.preprocessing(next_state))
                reward_sum += reward
               
                # Store the infomation in Replay Memory
                next_states = self.recent_states()

                if done:
                    self.replay.put(preprocessed_state, action_uelist, reward, None)
                else:
                    self.replay.put(preprocessed_state, action_uelist, reward, next_states)

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

    def optimize(self, gamma: float):

        if len(self.replay) < BATCH_SIZE:
            return

        # self.grouping_action.opt()
        # self.ue_action.opt()

        # Get Sample
        transitions = self.replay.sample(BATCH_SIZE)

        # Mask
        non_final_mask = torch.ByteTensor(list(map(lambda ns: ns is not None, transitions.next_state))).cuda()
        final_mask = 1 - non_final_mask

        state_batch: Variable = Variable(torch.cat(transitions.state).cuda())
        action_batch: Variable = Variable(torch.cat(transitions.action).cuda())
        reward_batch: Variable = Variable(torch.cat(transitions.reward).cuda())
        non_final_next_state_batch = Variable(torch.cat([ns for ns in transitions.next_state if ns is not None]).cuda())
        non_final_next_state_batch.volatile = True

        # Reshape States and Next States
        state_batch = state_batch.view([BATCH_SIZE, self.action_repeat, self.env.width, self.env.height])
        non_final_next_state_batch = non_final_next_state_batch.view([-1, self.action_repeat, self.env.width, self.env.height])
        non_final_next_state_batch.volatile = True

        # Clipping Reward between -2 and 2
        reward_batch.data.clamp_(-1, 1)

        # Predict by DQN Model
        q_pred = self.action_repeat()
        if self.mode == 'dqn':
            q_pred = self.dqn(state_batch)
        elif self.mode == 'lstm':
            q_pred, self.dqn_hidden_state, self.dqn_cell_state = self.dqn(state_batch, self.dqn_hidden_state,
                                                                          self.dqn_cell_state)

        q_values = q_pred.gather(1, action_batch)

        # Predict by Target Model
        target_values = Variable(torch.zeros(BATCH_SIZE, 1).cuda())
        if self.mode == 'dqn':
            target_pred = self.target(non_final_next_state_batch)
        elif self.mode == 'lstm':
            target_pred, self.target_hidden_state, self.target_cell_state = self.target(non_final_next_state_batch,
                                                                                        self.target_hidden_state,
                                                                                        self.target_cell_state)

        target_values[non_final_mask] = reward_batch[non_final_mask] + target_pred.max(1)[0] * gamma
        target_values[final_mask] = reward_batch[final_mask].detach()

        loss = F.smooth_l1_loss(q_values, target_values)

        # loss = torch.mean((target_values - q_values) ** 2)
        self.optimizer.zero_grad()
        loss.backward(retain_variables=True)

        if self.clip:
            for param in self.dqn.parameters():
                param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        reward_score = int(torch.sum(reward_batch).data.cpu().numpy()[0])
        q_mean = torch.sum(q_pred, 0).data.cpu().numpy()[0]
        target_mean = torch.sum(target_pred, 0).data.cpu().numpy()[0]

        return loss.data.cpu().numpy(), reward_score, q_mean, target_mean


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

