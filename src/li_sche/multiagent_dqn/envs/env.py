import json
import os
import argparse

from .ue import UE
from .constant import *
from ..agent.action_space import *
import random

## Constant
SLOT_PATTERN1       = ['D','D','S','U','U']
PATTERN_P1          = 5
SIMULATION_FRAME    = 200
NUMBER_OF_SUBFRAME  = 10
MAX_UPLINK_GRANT    = 124
MAX_GROUP           = 4


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
               slot_information,
               size_of_recv : int = 0, 
               size_of_exp : int = 0):
        
        reward = 0
        weight_recv = 1
        weight_exp = 0.8
        collision_weight = [1,1,1,1]
        success_weight = [1,1,1,1]

        if slot_information == 'D' or slot_information == 'S':
            return 0
        else :
            reward = (weight_recv * size_of_recv) - (weight_exp * size_of_exp)
        
        return reward



    def reset(self):
        return self.init()
    
    def send_DCI(self):
        return 
    

    # return S(t+1)
    def step(self, group_list : list):
        next_state = state()
        success_ul_uelist = list()
        failed_ul_uelist = list()
        slot_information = self.ran_config.slot_pattern[self.slot % self.ran_config.pattern_p]

        # match self.ran_config.slot_pattern[self.slot % self.ran_config.pattern_p] :
        #     case 'D':
        #         self.slot_type = 'D'
        #     case 'U':
        #         self.slot_type = 'S'
        #     case 'S':
        #         self.slot_type = 'D'

        exp_RB = 0
        recvd_RB = 0

        # Each UE in each group will randomly select a shared RB
        for i in range(len(group_list)):
            group : Gorup_result = group_list[i]
            nrofRB : int = group.get_RB()
            ul_uelist = group.get_ul_uelist


            rb_map = dict()
            for j in range(len(ul_uelist)) :

                # To calculate the expoected RB
                ue : UE = ul_uelist[j]
                exp_RB = exp_RB + ue.nrofRB

                # Each UE select a RB to transmit
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

        # Calculate the reward
        success_ul_ue : UE
        for success_ul_ue in success_ul_uelist:
            recvd_RB = recvd_RB + success_ul_ue.nrofRB

        reward = self.reward(slot_information = slot_information, size_of_recv = recvd_RB, size_of_exp=exp_RB, collision_list=failed_ul_uelist, success_list=success_ul_uelist)

        
        # The done condition
        global UE_list
        if self.flow >= SIMULATION_FRAME * NUMBER_OF_SUBFRAME* pow(2, self.ran_config.numerology):
            done = True

        if len(UE_list) == 0:
            done = True
            
                 
        ## Update the next round 
        self.slot = self.slot + 1
        done = False

        return next_state, reward, done, slot_information

        
        
        
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
        observation, reward, done, information = self.ran_system.step(group_list)
        return observation, reward, done, information

    def reset(self):
        """
        :return: observation array
        """
        state = self.ran_system.reset()
        return state