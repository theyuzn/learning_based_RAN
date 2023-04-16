'''
There are four action spaces
    1. Number of Group {1 ~ GROUP_MAX} -->  GROUP_MAX-dimension
    2. Number of shared RB {0 ~ (Total RB - 1)} --> (Total RB - 1)-dimention; 1 is reserved for URLLC
    3. For each UE, decide contention or not {schedule, contention} --> 2-dimension
    4. For each group, decide MCS index {1 ~ 27} --> 27-dimension
'''

from ..envs.ue import UE

# Decide the number of group [1 ~ MAX_GROUP]
class grouping_action_space:
    def __init__(self, action = -1, n = -1):
        self.action = action
        self.n = n
    
    def set_action(self, decision):
        self.action = decision
    
    def get_action(self):
        return self.action
    
    def get_dimension(self):
        return self.n


# Decide the number of shared RB
class rb_action_space:
    def __init__(self, action = -1, n = -1):
        self.action = action
        self.n = n

    def set_action(self, decision):
        self.action = decision

    def get_action(self):
        return self.action
    
    def get_dimension(self):
        return self.n
    
# Decide whether the UE should be scheduled or not
class ue_action_space:
    def __init__(self, action = -1, n = -1):
        self.action = action
        self.n = n

    def set_action(self, decision):
        self.action = decision

    def get_action(self):
        return self.action
    
    def get_dimension(self):
        return self.n
    
# Decide the MCS index
class mcs_action_space:
    def __init__(self, action = -1, n = -1):
        self.action = action
        self.n = n

    def set_action(self, decision):
        self.action = decision

    def get_action(self):
        return self.action
    
    def get_dimension(self):
        return self.n
    

class Gorup_result:
    def __init__(self, group_id = 0,ul_uelist = [], nrofRB = 0, mcs = 10):
        self.group_id = group_id
        self.ul_uelist = ul_uelist
        self.nrofRB = nrofRB
        self.mcs = mcs

    def add_UE(self, ue : UE):
        self.ul_uelist.appen(ue)

    def set_mcs(self, mcs : int):
        self.mcs = mcs

    def set_RB(self, nrofRB):
        self.nrofRB = nrofRB

    @property
    def get_id(self):
        return self.group_id

    @property
    def get_ul_uelist(self):
        return self.ul_uelist
    
    @property
    def get_RB(self):
        return self.nrofRB
    
    @property
    def get_mcs(self):
        return self.mcs