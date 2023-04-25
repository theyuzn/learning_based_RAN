import argparse
import math

from envs.ue import UE
from envs.env import State
from net.brain import *
from random import randrange

MAX_GROUP = 4   # 1 ~ 4 groups
MAX_MCS_INDEX = 29 # 0 ~ 28

class Resource:
    def __init__(self, group_id : int, mcs : int):
        self.group_id = 0
        self.mcs = 0

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
        self.clip: bool = args.clip
        self.seed: int = args.seed
        self.action_repeat: int = action_repeat
        self.frame_skipping: int = args.skip_action
        self._state_buffer = deque(maxlen=self.action_repeat)
        self.step = 0
        self.best_score = args.best or -10000
        self.best_count = 0
        self._play_steps = deque(maxlen=5)

        if cuda:
            self.device = torch.device("cuda")

        # torch init
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        np.random.seed(args.seed)
        self.epsilon = EPSILON_START


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


    def select_action(self, state: State, step) -> Joint_Action:
        self.step = step
        joint_action = Joint_Action()
        self.epsilon = EPSILON_END + (EPSILON_START - EPSILON_END) * math.exp(-1. * self.step / EPSILON_DECAY)

        resources = []
        # Randomly select a grouping number
        if self.epsilon > random():
            # Group number
            nrofGroup = randrange(MAX_GROUP)# 1 ~ 4

            # UE
            for i in range(len(state.ul_uelist)):
                group_id = randrange(nrofGroup)
                state.ul_uelist[i].set_group(group_id)

            # MCS
            for i in range(nrofGroup):
                mcs = randrange(MAX_MCS_INDEX)
                res = Resource(group_id=i, mcs=mcs)
                resources.append(mcs)
            
            joint_action.set_action(nrofGroup=nrofGroup, resource_allocation=resources, ul_uelist=state.ul_uelist)
            return joint_action
        
        else:
            processed_state = self.preprocessing(state=state)
            # Group number
            state_tensor = torch.as_tensor(processed_state, dtype = torch.float)
            action_prob = self.resource_policy_net.forward(state = state_tensor)
            nrofGroup = torch.multinomial(action_prob, 1) + 1   # 1 ~ 4

            # UE
            for i in range(len(state.ul_uelist)):
                processed_state = np.append(processed_state, nrofGroup)
                ue_processed_state = np.append(processed_state, state.ul_uelist[i].sizeOfData)
                ue_processed_state = np.append(ue_processed_state, state.ul_uelist[i].delay_bound)
                ue_processed_state = np.append(ue_processed_state, state.ul_uelist[i].errorrate)
                ue_state_tensor = torch.as_tensor(ue_processed_state, dtype = torch.float)
                ue_action = self.ue_policy_net.forward(state = ue_state_tensor)
                group_id = torch.multinomial(ue_action, 1) # 0 ~ nrofGroup - 1
                while group_id > nrofGroup - 1:
                    group_id = torch.multinomial(ue_action, 1) # 0 ~ nrofGroup - 1
                state.ul_uelist[i].set_Group(group_id)

            # MCS
            for i in range(nrofGroup):
                mcs_state_tensor = torch.as_tensor(processed_state, dtype = torch.float)
                mcs_action = self.mcs_policy_net.forward(state = mcs_state_tensor)
                mcs = torch.multinomial(mcs_action, 1)
                resources.append(mcs)

            joint_action.set_action(nrofGroup= nrofGroup, resource_allocation= resources, ul_uelist= state.ul_uelist)
            return joint_action
                
            
    def optimization(self,  gamma: float = 0.99):
        return




