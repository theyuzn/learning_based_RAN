import argparse
import math
import torch
import logging
import copy
from random import random, sample
from collections import namedtuple, deque
import numpy as np
import torch.optim as optim

from .envs.ue import UE
from .envs.env import State
from .net.brain import Regression_DQN
from random import randrange
from .envs.env import RAN_system
from .utils.constants import *

# Logging
logger = logging.getLogger('DQN')
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(message)s')
file_handler = logging.FileHandler(f'dqn_{STEP}_{MODE_R}.log')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


class ReplayMemory(object):
    def __init__(self, capacity=REPLAY_MEMORY):
        self.capacity = capacity
        self.memory = deque([], maxlen=self.capacity)
        self.Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state'))
        self._available = False

    def push(self, state: torch.Tensor, action: torch.Tensor, reward: torch.Tensor, next_state: torch.Tensor):
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

'''
** Agent
This agent is to decide the number of shared RB
1. The ratio of shared RB is defined as N_shared_RB/N_Total_RB
2. Use the ration of shared RB to determine the threshold of data and delay
3. The Agent is to assist the scheduler speed up the schduling
    where if UE is satisfying the threshold, it will be well-scheduled,
    otherwise, the UE will be arranged to perform contention in shared resources.
'''
class Agent():
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.seed: int = args.seed
        self.action_repeat: int = args.repeat
        
        self.device = torch.device("cpu")

        # torch init
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        np.random.seed(args.seed)

        # Common
        self.memory = ReplayMemory(capacity=50000)
        self.epsilon = EPSILON_START

        # Environment
        self.env = RAN_system(args=args)
        self.step = 0

        # DQN Model
        self.shared_rb_policy_net = Regression_DQN(MAX_RB).to(self.device)
        self.shared_rb_target_net = Regression_DQN(MAX_RB).to(self.device)
        self.shared_rb_target_net.load_state_dict(self.shared_rb_policy_net.state_dict())

        # Optimizer
        self.shared_rb_optimizer = optim.AdamW(self.shared_rb_policy_net.parameters(), lr=LEARNING_RATE, amsgrad = True)

    ########################## Preprocessing ##########################
    def preprocessing(self, state : State):
        if len(state.ul_uelist) < 1:
            return np.empty(0)
        
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
    
    def sharedRB_select_action(self, state : np.ndarray):
        self.epsilon = EPSILON_END + (EPSILON_START - EPSILON_END) * math.exp(-1. * self.step / EPSILON_DECAY)

        if self.epsilon > random():
            nrofSharedRB = randrange(MAX_RB) # 0 ~ MAX_RB - 1
            return torch.as_tensor([[nrofSharedRB]]).to(self.device)
        else:
            state_tensor = torch.as_tensor([state], dtype = torch.float).to(self.device)
            action_prob = self.shared_rb_policy_net.forward(state = state_tensor)
            nrofSharedRB = torch.multinomial(action_prob, 1)   # 0 ~ MAX_RB - 1
            return nrofSharedRB
        
    
    def pretrain(self):
        
        return
    

    def train(self, gamma : float = 0.99):
        # Initial States
        reward_sum = 0.
        q_mean = [0., 0.]
        target_mean = [0., 0.]

        while True:
            ### states is an np.stack which stores preprocessed state -- np.ndarray
            # states : np.ndarray = self.get_initial_states()
            state : State =  self.env.reset()
            losses = []
            target_update_flag = False
            done = False

            while True:   
                ## state is used to check which the schedule slot is. Ex: 'D', 'U', 'S'
                ## preprocessed_state is processed state, type : np.ndarray
                schudule_slot_info = state.get_schedule_slot_info()
                ul_uelist = state.ul_uelist
                state = self.preprocessing(state)
                action = torch.tensor([[0]]).to(self.device)

                match schudule_slot_info:
                    case 'D':
                        # skip
                        self.DL_slot()

                    case 'S':
                        # skip
                        self.Special_slot()

                    case 'U':
                        nrofSharedRB = 0
                        if len(ul_uelist) > 0:
                            nrofSharedRB = self.sharedRB_select_action(state = state)
                        action = nrofSharedRB
                        nrofScheRB = MAX_RB - nrofSharedRB
                        if nrofSharedRB == 0:
                            for i in range(len(ul_uelist)):
                                ul_uelist[i].set_Group(0)
                        else:
                            shared_cnt = 0
                            sche_cnt = 0
                            for i in range(len(ul_uelist)):
                                ul_uelist[i].set_RB(1)
                                if shared_cnt <= nrofSharedRB and sche_cnt < nrofScheRB:
                                    if ul_uelist[i].delay_bound <= 30:
                                        ul_uelist[i].set_Group(0)
                                        sche_cnt += 1
                                    else:
                                        ul_uelist[i].set_Group(1)
                                        shared_cnt += 1
                                elif shared_cnt <= nrofSharedRB:
                                    ul_uelist[i].set_Group(1)
                                    shared_cnt += 1
                                else:
                                    ul_uelist[i].set_Group(0)
                                    sche_cnt += 1

                        # Increase step 
                        # Training step is following the UL
                        # To decay the epsilon
                        self.step += 1
            
                next_state, reward, done = self.env.step(ul_uelist)

                if len(state) > 0:
                    if done:
                        self.memory.push( torch.as_tensor(state, dtype = torch.float), 
                                         action, 
                                         torch.tensor([reward]), 
                                         None)
                    else:
                        self.memory.push( torch.as_tensor(state, dtype = torch.float), 
                                         action, 
                                         torch.tensor([reward]),  
                                         torch.as_tensor(self.preprocessing(next_state), dtype=torch.float))
                           
                # Change States
                state = next_state

                # Optimize
                if self.memory.is_available():
                    loss, reward_sum, q_mean, target_mean = self.optimize(gamma = gamma)
                    losses.append(loss)

                # Soft update of the target network's weights
                # θ′ ← τ θ + (1 −τ )θ′
                target_net_state_dict = self.shared_rb_target_net.state_dict()
                policy_net_state_dict = self.shared_rb_policy_net.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
                self.shared_rb_target_net.load_state_dict(target_net_state_dict)


                if self.step % TARGET_UPDATE_INTERVAL == 0:
                    self.shared_rb_target_net = copy.deepcopy(self.shared_rb_policy_net)

                if done:
                    print("Done!!")
                    break

            # Logging
            mean_loss = np.mean(losses)
            target_update_msg = '[target updated]' if target_update_flag else ''
            logger.info(f'[{self.step}] Loss:{mean_loss:<8.4}'  
                        f'  RewardSum:{reward_sum:<3} Q:[{q_mean:<6.4}] '
                        f'  T:[{target_mean:<6.4}] '
                        f'  Epsilon:{self.epsilon:<6.4}{target_update_msg}')                
                
            
    def optimize(self,  gamma: float = 0.99):

        # Get Samples : return Transition(*zip(transitions))
        batch = self.memory.sample(BATCH_SIZE)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device = self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None]).to(self.device)
        state_batch = torch.cat(batch.state).to(self.device)
        action_batch = torch.cat(batch.action).to(self.device)
        reward_batch = torch.cat(batch.reward).to(self.device)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_batch = state_batch.view([BATCH_SIZE, 12])
        state_action_values = self.shared_rb_policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(BATCH_SIZE, device=self.device)
        with torch.no_grad():
            non_final_next_states = non_final_next_states.view([-1, 12])
            next_state_values[non_final_mask] = self.shared_rb_target_net(non_final_next_states).max(1)[0]
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * gamma) + reward_batch

        # Compute Huber loss
        criterion = torch.nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.shared_rb_optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.shared_rb_policy_net.parameters(), 100)
        self.shared_rb_optimizer.step()

        reward_score = int(torch.sum(reward_batch).data.cpu().numpy())
        q_mean = torch.sum(state_action_values, 0).data.cpu().numpy()[0]
        target_mean = torch.sum(next_state_values, 0).data.cpu().numpy()

        return loss.data.cpu().numpy(), reward_score, q_mean, target_mean




