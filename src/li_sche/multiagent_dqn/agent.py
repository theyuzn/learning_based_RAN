import argparse
import math
import torch
import logging


from .envs.ue import UE
from .envs.env import State
from .net.brain import *
from random import randrange
from .envs.env import Env


MAX_GROUP = 4   # 1 ~ 4 groups
MAX_MCS_INDEX = 29 # 0 ~ 28
MAX_RB = 248

# Logging
logger = logging.getLogger('DQN')
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(message)s')

file_handler = logging.FileHandler(f'dqn.log')
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


class Resource:
    def __init__(self, group_id : int, mcs : int):
        self.group_id = group_id
        self.mcs = mcs

class Resource_Action:
    def __init__(self):
        self.action = list(None)*MAX_GROUP
        self.n = MAX_GROUP
    
    def get_dim(self):
        return self.n

    def nrofGroup(self):
        return len(self.action)
    
    def set_Group(self, nrofGroup):
        for i in range(nrofGroup):
            self.action[i].group_id = i
            self.action[i].mcs = 10

    def set_mcs(self, group_id, mcs):
        self.action[group_id].mcs = mcs

class UE_Action:
    def __init__(self):
        self.action = 0
        self.n = MAX_GROUP
    
    def get_dim(self):
        return self.n

    def set_group(self, group_id):
        self.action = group_id
    
    def get_group(self):
        return self.action
    
class MCS_Action:
    def __init__(self):
        self.action = 0
        self.n = MAX_MCS_INDEX
    
    def get_dim(self):
        return self.n

    def set_mcs(self, mcs):
        self.action = mcs
    
    def get_group(self):
        return self.action


class Joint_Action:
    def __init__(self):
        self.nrofGroup = 0
        self.resource = []
        self.ul_uelist = []

    def set_action(self, nrofGroup : int, resource_allocation = [], ul_uelist = []):
        self.nrofGroup = nrofGroup
        self.resource = resource_allocation
        self.ul_uelist = ul_uelist


'''
** Coorperative Multi-Agent DQN
This study uses two agents : 1. Resource_Agnet 2. UE_Agent
1. Resource_Agent   : decide the resource arragement ( and mcs TODO ...)
2. UE_Agent         : decide if the UE is scheduled or not and allocate the UE to appropriate resource.
'''
class Agent():
    def __init__(self, args: argparse.Namespace, cuda = True, action_repeat: int = 4):
        self.args = args
        self.seed: int = args.seed
        self.action_repeat: int = action_repeat
        
        self.device = torch.device("cuda")

        # torch init
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        np.random.seed(args.seed)

        # Common
        # self.resource_memory = ReplayMemory(capacity = 50000)
        # self.ue_memory = ReplayMemory(capacity = 50000)
        # self.mcs_memory = ReplayMemory(capacity = 50000)
        self.memory = ReplayMemory(capacity=50000)
        self.epsilon = EPSILON_START

        # Environment
        self.env = Env(args=args)
        self.step = 0

        # Action space
        # self.resource_action = Resource_Action()
        # self.ue_action = UE_Action()
        # self.mcs_action = MCS_Action()

        # DQN Model
        self.shared_rb_policy_net = Shrared_RB_DQN(MAX_RB).to(self.device)
        self.shared_rb_target_net = Shrared_RB_DQN(MAX_RB).to(self.device)
        self.shared_rb_target_net.load_state_dict(self.shared_rb_policy_net.state_dict())
        # self.shared_policy_net      = Shared_DQN().to(self.device)
        # self.shared_target_net      = Shared_DQN().to(self.device)
        # self.resource_policy_net    = Resource_DQN(self.action_space.get_dim()).to(self.device)
        # self.resource_target_net    = Resource_DQN(self.action_space.get_dim()).to(self.device)
        # self.ue_policy_net          = UE_Classification_DQN(self.ue_action.get_dim()).to(self.device)
        # self.ue_target_net          = UE_Classification_DQN(self.ue_action.get_dim()).to(self.device)
        # self.mcs_policy_net         = MCS_DQN(self.mcs_action.get_dim()).to(self.device)
        # self.mcs_target_net         = MCS_DQN(self.mcs_action.get_dim()).to(self.device)

        # Optimizer
        self.shared_rb_optimizer = optim.AdamW(self.shared_rb_policy_net.parameters(), lr=LEARNING_RATE, amsgrad = True)
        # self.resource_optimizer = optim.AdamW(self.resource_policy_net.parameters(), lr=LEARNING_RATE, amsgrad = True)
        # self.ue_optimizer = optim.AdamW(self.ue_policy_net.parameters(), lr=LEARNING_RATE, amsgrad = True)
        # self.mcs_optimizer = optim.AdamW(self.mcs_policy_net.parameters(), lr=LEARNING_RATE, amsgrad = True)


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
    
    # input processed state with 12 variable
    def resource_select_action(self, state : np.ndarray):
        if self.epsilon > random():
            nrofGroup = randrange(MAX_GROUP)# 1 ~ 4
            return nrofGroup
        else:
            state_tensor = torch.as_tensor(state, dtype = torch.float)
            action_prob = self.resource_policy_net.forward(state = state_tensor)
            nrofGroup = torch.multinomial(action_prob, 1) + 1   # 1 ~ 4
            return nrofGroup

    def ue_select_action(self, state : np.ndarray):
        if self.epsilon > random():
            group_id = randrange(state[12])
            return group_id
        else:
            state_tensor = torch.as_tensor(state, dtype = torch.float)
            action_prob = self.ue_policy_net.forward(state = state_tensor)
            group_id = torch.multinomial(action_prob, 1)
            return group_id
    
    def mcs_select_action(self, state : np.ndarray):
        if self.epsilon > random():
            mcs = randrange(MAX_MCS_INDEX)
            return mcs
        else:
            state_tensor = torch.as_tensor(state, dtype = torch.float)
            action_prob = self.mcs_policy_net.forward(state = state_tensor)
            mcs = torch.multinomial(action_prob, 1)
            return mcs
        

    def train(self, gamma : float = 0.99):
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


            previous_ue_state : np.ndarray
            previous_ue_state_id : int
            previous_mcs_state : np.ndarray
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

                # Increase step
                self.step += 1

                if done:
                    break

            # Logging
            mean_loss = np.mean(losses)
            target_update_msg = '[target updated]' if target_update_flag else ''
            logger.info(f'[{self.step}] Loss:{mean_loss:<8.4}'  
                        f'RewardSum:{reward_sum:<3} Q:[{q_mean:<6.4}] '
                        f'T:[{target_mean:<6.4}] '
                        f'Epsilon:{self.epsilon:<6.4}{target_update_msg}')                
                
            
    def optimize(self,  gamma: float = 0.99):

        # Get Samples : return Transition(*zip(transitions))
        batch = self.memory.sample(BATCH_SIZE)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=self.device, dtype=torch.bool)
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
        criterion = nn.SmoothL1Loss()
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




