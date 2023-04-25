from envs.ue import UE

MAX_GROUP = 4   # 1 ~ 4 groups

class Resource:
    def __init__(self):
        self.group_id = 0
        self.mcs = 0

class Resource_Action:
    def __init__(self):
        self.action = list(Resource)*MAX_GROUP
        self.action = 0
        self.n = MAX_GROUP

    def nrofGroup(self):
        return len(self.action)
    
    def set_Group(self, nrofGroup):
        for i in range(nrofGroup):
            self.action[i].group_id = i
            self.action[i].mcs = 10

    def set_mcs(self, group_id, mcs):
        self.action[group_id].mcs = mcs


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
class Resource_Agent():
    def __init__(self):
        self.policy_net = DQN()
        self.target_net = DQN()
        self.action = Resource_Action()


class UE_Agent():
    def __init__(self):
        self.policy_net = DQN()
        self.target_net = DQN()
        self.action = UE_Action()



