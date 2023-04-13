import json
import os
import argparse

from .ue import UE
from .constant import *
from ..agent.action_space import *

SLOT_PATTERN1 = ['D','D','S','U','U']
PATTERN_P1 = 5

def decode_json(dct):
    return  UE(id = dct[ID],
               sizeOfData = dct[SIZE],
               delay_bound = dct[DELAY],
               type = dct[TYPE])

class RAN_config:
    def __init__(self, 
                 BW         = 40, 
                 numerology = 1, 
                 nrofRB     = 106, 
                 k0         = 0,
                 k1         = 2,
                 k2         = 4,
                 slot_pattern = SLOT_PATTERN1,
                 pattern_p = PATTERN_P1):
        self.BW             = BW
        self.numerology     = numerology
        self.nrofRB         = nrofRB
        self.k0             = k0
        self.k1             = k1
        self.k2             = k2
        self.slot_pattern   = slot_pattern
        self.pattern_p      = pattern_p


MAX_GROUP = 4
 
class RAN_system:
    def __init__(self, args : argparse.Namespace):
        self.slot = 0
        self.uelist = []
        self.args = args
        self.ran_config = RAN_config(BW = self.args.bw,
                                     numerology = self.args.mu,
                                     nrofRB = self.args.rb,
                                     k0 = self.args.k0,
                                     k1 = self.args.k1,
                                     k2 = self.args.k2,
                                     )
    
    def init(self):
        self.slot = 0
        self.uelist = []
        self.collision = [None]*MAX_GROUP
        cur_path = os.path.dirname(__file__)
        new_path = os.path.join(cur_path, '../../../data/uedata.json')

        with open(new_path, 'r') as ue_file:
            self.uelist = json.load(ue_file,object_hook=decode_json) 


    def reset(self):
        self.init()
        self.ul_uelist = []
        self.collision = 0
        return self.ul_uelist, self.collision
    

    def step(self, group_number, shared_rb_number, ul_action):
        print("TODO")

        
        
class Env:
    def __init__(self, args : argparse.Namespace):
        self.args = args
        self.ran_system = RAN_system(self.args)
            
    # return {ul_uelist, collision}
    def init(self):
        observation = self.ran_system.reset()
        return self.ran_system.reset()

    def step(self, 
             group_number : grouping_action_space, 
             shared_rb_number : rb_action_space, 
             ul_action : list):
        observation, reward, done, info = self.ran_system.step(group_number, shared_rb_number, ul_action)
        return observation, reward, done, info

    def reset(self):
        """
        :return: observation array
        """
        observation = self.ran_system.reset()
        return observation