import argparse
import math

from envs.ue import UE
from envs.env import State
from net.brain import *
from random import randrange
from envs.env import Env

MAX_GROUP = 4   # 1 ~ 4 groups
MAX_MCS_INDEX = 29 # 0 ~ 28

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
        if cuda:
            self.device = torch.device("cuda")

        # torch init
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        np.random.seed(args.seed)

        # Common
        self.resource_memory = ReplayMemory(capacity = 50000)
        self.ue_memory = ReplayMemory(capacity = 50000)
        self.mcs_memory = ReplayMemory(capacity = 50000)
        self.epsilon = EPSILON_START

        # Environment
        self.env = Env(args=args)
        self.step = 0

        # Action space
        self.resource_action = Resource_Action()
        self.ue_action = UE_Action()
        self.mcs_action = MCS_Action()

        # DQN Model
        # self.shared_policy_net      = Shared_DQN().to(self.device)
        # self.shared_target_net      = Shared_DQN().to(self.device)
        self.resource_policy_net    = Resource_DQN(self.action_space.get_dim()).to(self.device)
        self.resource_target_net    = Resource_DQN(self.action_space.get_dim()).to(self.device)

        self.ue_policy_net          = UE_Classification_DQN(self.ue_action.get_dim()).to(self.device)
        self.ue_target_net          = UE_Classification_DQN(self.ue_action.get_dim()).to(self.device)

        self.mcs_policy_net         = MCS_DQN(self.mcs_action.get_dim()).to(self.device)
        self.mcs_target_net         = MCS_DQN(self.mcs_action.get_dim()).to(self.device)

        # Optimizer
        self.resource_optimizer = optim.AdamW(self.resource_policy_net.parameters(), lr=LEARNING_RATE, amsgrad = True)
        self.ue_optimizer = optim.AdamW(self.ue_policy_net.parameters(), lr=LEARNING_RATE, amsgrad = True)
        self.mcs_optimizer = optim.AdamW(self.mcs_policy_net.parameters(), lr=LEARNING_RATE, amsgrad = True)


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
                joint_action : Joint_Action = Joint_Action()
                

                match schudule_slot_info:
                    case 'D':
                        # skip
                        joint_action = self.DL_slot()
                    case 'S':
                        # skip
                        joint_action = self.Special_slot()
                    case 'U':
                        self.epsilon = EPSILON_END + (EPSILON_START - EPSILON_END) * math.exp(-1. * self.step / EPSILON_DECAY)

                        # Resource
                        nrofGroup = self.resource_select_action(state = state)

                        # UE
                        ue_state = np.append(state, nrofGroup)
                        sample_id = randrange(len(ul_uelist))
                        group_data = np.zeros(nrofGroup)
                        group_delay_min = np.zeros(nrofGroup)
                        for i in range(len(ul_uelist)):
                            each_ue_state = np.append(ue_state, ul_uelist[i].sizeOfData)
                            each_ue_state = np.append(each_ue_state, state.ul_uelist[i].delay_bound)
                            each_ue_state = np.append(each_ue_state, state.ul_uelist[i].errorrate)

                            if i == sample_id:
                                previous_ue_state = each_ue_state
                                previous_ue_state_id = ul_uelist[i].id

                            group_id = self.ue_select_action(each_ue_state)
                            ul_uelist[i].set_Group(group_id)

                            group_data[group_id] += ul_uelist[i].sizeOfData
                            if group_delay_min[group_id] > ul_uelist[i].delay_bound or group_delay_min[group_id] == 0.:
                                group_delay_min[group_id] = ul_uelist[i].delay_bound
                                
                        # MCS
                        mcs_state = np.append(state, nrofGroup)
                        resources = np.zeros(nrofGroup)
                        for i in range(nrofGroup):
                            each_mcs_state = np.append(mcs_state, group_data[i])
                            each_mcs_state = np.append(each_mcs_state, group_delay_min[i])
                            mcs = self.mcs_select_action(state= each_mcs_state)
                            resources[i] = mcs


                        
                        joint_action = self.select_action(state=state)
                
                next_state, reward, done = self.env.step(joint_action)
                
                if done:
                    self.memory.push(state, joint_action, reward, None)
                else:
                    self.memory.push(state, joint_action, reward, next_state)
                
                reward_sum += reward
           
                # Change States
                state = next_state
                

                # Optimize
                if self.memory.is_available():
                    loss, reward_sum, q_mean, target_mean = self.optimize(gamma = gamma)
                    # losses.append(loss[0])

                if done:
                    break

                # Increase step
                self.step += 1
                play_steps += 1             
            
            break      
                
            
    def optimize(self,  gamma: float = 0.99):

        # Get Samples : return Transition(*zip(transitions))
        batch = self.memory.sample(BATCH_SIZE)
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




