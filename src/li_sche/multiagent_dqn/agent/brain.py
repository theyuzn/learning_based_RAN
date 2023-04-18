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


from ..envs.env import Env, MAX_GROUP
from ..algorithm.net import *
from ..algorithm.constant import *
from .group_agent import GroupAgent
from .rb_action import RBAction
from .ue_action import UEAgent

class Brain():
    # Init
    def __init__(self, args: argparse.Namespace, cuda = True, action_repeat: int = 4):

        # Environment
        self.env = Env(args=args)

        # Agents
        self.grouping_action = GroupAgent(args=args)
        # self.rb_action = RBAction(args=args)
        # self.ue_action = UEAgent(args=args)


    def get_initial_states(self):
        state = self.env.reset()
        states = np.stack([state for _ in range(self.action_repeat)], axis=0)
        self._state_buffer = deque(maxlen=self.action_repeat)
        for _ in range(self.action_repeat):
            self._state_buffer.append(state)
        return states

    def add_state(self, state):
        self._state_buffer.append(state)

    def recent_states(self):
        return np.array(self._state_buffer)

    ############# Test #############
    def test_system(self):
        state = self.env.reset()
        slot = 0
        done = False
        while not done:
            ul_uelist = state.ul_uelist

            for i in range(len(ul_uelist)):
                ul_uelist[i].set_Group(randrange(MAX_GROUP))
                ul_uelist[i].set_RB(1)

            state, reward, done = self.env.step(action = ul_uelist)

            print(f'slot:\t${slot}\t==>,len(state):\t{len(state.ul_uelist)}\t,reward:\t{reward}\t,done:\t{done}')
            slot += 1



    ############# Train #############
    def train(self, gamma: float = 0.99):
        # Initial States
        reward_sum = 0.
        q_mean = [0., 0.]
        target_mean = [0., 0.]

        while True:
            # Init LSTM States
            # if self.mode == 'lstm':
            #     # For Training
            #     self.train_hidden_state, self.train_cell_state = self.dqn.reset_states(self.train_hidden_state,self.train_cell_state)

            states = self.get_initial_states()
            losses = []
            target_update_flag = False
            play_flag = False
            play_steps = 0
            real_play_count = 0
            real_score = 0
            reward = 0
            done = False

            while True:
                # Get Action
                grouping_action: torch.LongTensor = self.grouping_action.select_action(states)

                for _ in range(self.frame_skipping):
                    # step 에서 나온 observation은 버림
                    observation, reward, done, info = self.env.step(action[0, 0])
                    next_state = self.env.get_screen()
                    self.add_state(next_state)

                    if done:
                        break

                # Store the infomation in Replay Memory
                next_states = self.recent_states()
                if done:
                    self.replay.put(states, action, reward, None)
                else:
                    self.replay.put(states, action, reward, next_states)

                # Change States
                states = next_states

                # Optimize
                if self.replay.is_available():
                    loss, reward_sum, q_mean, target_mean = self.optimize(gamma)
                    losses.append(loss[0])

                if done:
                    break

                # Increase step
                self.step += 1
                play_steps += 1

                # Target Update
                if self.step % TARGET_UPDATE_INTERVAL == 0:
                    self._target_update()
                    target_update_flag = True


                # Play
                if self.step % PLAY_INTERVAL == 0:
                    play_flag = True

                    scores = []
                    counts = []
                    for _ in range(PLAY_REPEAT):
                        score, real_play_count = self.play(logging=False, human=False)
                        scores.append(score)
                        counts.append(real_play_count)
                        logger.debug(f'[{self.step}] [Validation] play_score: {score}, play_count: {real_play_count}')
                    real_score = int(np.mean(scores))
                    real_play_count = int(np.mean(counts))

                    if self.best_score <= real_score:
                        self.best_score = real_score
                        self.best_count = real_play_count
                        logger.debug(f'[{self.step}] [CheckPoint] Play: {self.best_score} [Best Play] [checkpoint]')
                        self.save_checkpoint(
                            filename=f'dqn_checkpoints/chkpoint_{self.mode}_{self.best_score}.pth.tar')

            self._play_steps.append(play_steps)

            # Play
            if play_flag:
                play_flag = False
                logger.info(f'[{self.step}] [Validation] mean_score: {real_score}, mean_play_count: {real_play_count}')

            # Logging
            mean_loss = np.mean(losses)
            target_update_msg = '  [target updated]' if target_update_flag else ''
            # save_msg = '  [checkpoint!]' if checkpoint_flag else ''
            logger.info(f'[{self.step}] Loss:{mean_loss:<8.4} Play:{play_steps:<3}  '  # AvgPlay:{self.play_step:<4.3}
                        f'RewardSum:{reward_sum:<3} Q:[{q_mean[0]:<6.4}, {q_mean[1]:<6.4}] '
                        f'T:[{target_mean[0]:<6.4}, {target_mean[1]:<6.4}] '
                        f'Epsilon:{self.epsilon:<6.4}{target_update_msg}')

    def optimize(self, gamma: float):
        if self.mode == 'lstm':
            # For Optimization
            self.dqn_hidden_state, self.dqn_cell_state = self.dqn.reset_states(self.dqn_hidden_state,
                                                                               self.dqn_cell_state)
            self.target_hidden_state, self.target_cell_state = self.dqn.reset_states(self.target_hidden_state,
                                                                                     self.target_cell_state)

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
        non_final_next_state_batch = non_final_next_state_batch.view(
            [-1, self.action_repeat, self.env.width, self.env.height])
        non_final_next_state_batch.volatile = True

        # Clipping Reward between -2 and 2
        reward_batch.data.clamp_(-1, 1)

        # Predict by DQN Model
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

    def _target_update(self):
        self.target = copy.deepcopy(self.dqn)

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

