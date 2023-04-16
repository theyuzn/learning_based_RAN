import json
import os
import argparse

from .ue import UE
from .constant import *
from ..agent.action_space import *
import random

SLOT_PATTERN1 = ['D','D','S','U','U']
PATTERN_P1 = 5
## The global variable to store all the UE info in system ##
UE_list = []

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


class state :
    def __init__(self, 
                 ul_uelist = [], 
                 collision_list = [None]*MAX_GROUP, 
                 success_list = [None]*MAX_GROUP):
        self.ul_uelist = ul_uelist
        self.collision_list = collision_list
        self.success_list = success_list

    def add_ul_ue(self, ue : UE):
        self.ul_uelist .append(ue)

    def add_collision(self, group_id : int, nrofCollision = 1):
        self.collision_list[group_id - 1] = self.collision_list[group_id - 1] + nrofCollision

    def add_success(self, group_id : int):
        self.success_list[group_id] = self.success_list[group_id] + 1

    def reset(self):
        self.ul_uelist = []
        self.collision_list = [None]*MAX_GROUP
 
class RAN_system:
    def __init__(self, args : argparse.Namespace):
        self.slot = 0
        self.args = args
        self.ran_config = RAN_config(BW = self.args.bw,
                                     numerology = self.args.mu,
                                     nrofRB = self.args.rb,
                                     k0 = self.args.k0,
                                     k1 = self.args.k1,
                                     k2 = self.args.k2,
                                     )
    
    def init(self):
        global UE_list

        self.slot = 0
        UE_list = []
        self.state = state()
        cur_path = os.path.dirname(__file__)
        new_path = os.path.join(cur_path, '../../../data/uedata.json')

        with open(new_path, 'r') as ue_file:
            UE_list = json.load(ue_file,object_hook=decode_json) 
        print(UE_list)
        return self.state

    def reward(self, 
               size_of_recv : int = 0, 
               size_of_exp : int = 0, 
               collision_list = [None]*MAX_GROUP, 
               success_list = [None]*MAX_GROUP):
        collision_weight = [1,1,1,1]
        success_weight = [1,1,1,1]



    def reset(self):
        return self.init()
    

    # return S(t+1)
    def step(self, group_list : list):
        slot = slot + 1
        next_state = state()
        success_ul_uelist = list()
        failed_ul_uelist = list()

        # Each UE in each group will randomly select a shared RB
        for i in range(len(group_list)):
            group : Gorup_result = group_list[i]
            nrofRB : int = group.get_RB()
            ul_uelist = group.get_ul_uelist

            # Randomly select
            rb_map = dict()
            for j in range(len(ul_uelist)) :
                rb_id = random.randrange(nrofRB) + 1
                ul_uelist[j].set_RB_ID(rb_id)

                if rb_id in rb_map:
                    rb_map[rb_id].append(ul_uelist[j])
                else:
                    rb_map[rb_id] = [ul_uelist[j]]

            # Contention resolution
            for id in rb_map:
                if len(rb_map[id]) > 1:
                    next_state.add_collision(group.get_id(), len(rb_map[id]))
                    
                    for ue_id in range(len(rb_map[id])):
                        failed_ul_uelist.append(rb_map[id][ue_id])

                elif  len(rb_map[id]) == 1:
                    next_state.add_success(group.get_id())
                    success_ul_uelist.append(rb_map[id][0])
                    
                else :
                    continue
            
                
            


        done = False

        
        
        
class Env:
    def __init__(self, args : argparse.Namespace):
        self.args = args
        self.ran_system = RAN_system(self.args)
            
    # return state with two observations {ul_uelist, collision}
    def init(self):
        global UE_list
        self.ran_system.reset()
        print(UE_list)
        return 

    def step(self, group_list : list):
        observation, reward, done = self.ran_system.step(group_list)
        return observation, reward, done

    def reset(self):
        """
        :return: observation array
        """
        state = self.ran_system.reset()
        return state