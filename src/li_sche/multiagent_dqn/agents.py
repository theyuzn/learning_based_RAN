import argparse

from envs.ue import UE
from net.brain import *

MAX_GROUP = 4   # 1 ~ 4 groups

class Resource:
    def __init__(self):
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


class Joint_Action:
    def __init__(self):
        self.nrofGroup = 0
        self.resource = list(Resource)
        self.ul_uelist = list(UE)

    def set_action(self, nrofGroup : int, resource_allocation : list(Resource), ul_uelist : list(UE)):
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

        # torch init
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        np.random.seed(args.seed)

        # Action space
        self.resource_action = Resource_Action()
        self.ue_action = UE_Action()

        # DQN Model
        self.resource_policy_net = Resource_DQN(self.action_space.get_dim())
        self.resource_target_net = Resource_DQN(self.action_space.get_dim())
        self.ue_policy_net = UE_DQN(self.ue_action.get_dim())
        self.ue_target_net = UE_DQN(self.ue_action.get_dim())
        
        # self.net.cuda()
        self.target: Decide_Grouping = copy.deepcopy(self.net)
        self.optimizer = optim.Adam(self.net.parameters(), lr=LEARNING_RATE)
        self.epsilon = EPSILON_START





