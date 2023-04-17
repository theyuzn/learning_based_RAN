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
SLOT_PATTERN2       = ['D','D','D','S','U']
PATTERN_P2          = 5
SLOT_PATTERN3       = ['D','D','D','D','D','D','D', 'S','U','U']
PATTERN_P3          = 10

SIMULATION_FRAME    = 200
NUMBER_OF_SUBFRAME  = 10
MAX_UPLINK_GRANT    = 124
MAX_GROUP           = 4
PRE_SCHE_SLOT       = 6
SIZE_PER_RB         = 400


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
                 schedule_slot_info = 'D',
                 ul_uelist = [], 
                 collision_list = [None]*MAX_GROUP, 
                 success_list = [None]*MAX_GROUP):
        
        self.schedule_slot_info = schedule_slot_info
        self.ul_uelist = ul_uelist
        self.collision_list = collision_list
        self.success_list = success_list

    def rm_ue(self, ue : UE):
        self.ul_uelist.remove(ue)

    def set_schedule_slot_info(self, schedule_slot_info):
        self.schedule_slot_info = schedule_slot_info

    def reset(self):
        global UE_list
        self.ul_uelist = UE_list
        self.collision_list = [None]*MAX_GROUP
        self.success_list = [None]*MAX_GROUP
        self.schedule_slot_info = 'D'
 

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

        ## To add UE data to global variable UE_list
        global UE_list
        self.slot = 0
        UE_list = []
        cur_path = os.path.dirname(__file__)
        new_path = os.path.join(cur_path, '../../../data/uedata.json')
        with open(new_path, 'r') as ue_file:
            UE_list = json.load(ue_file,object_hook=decode_json) 

    ### Return 'D'; 'S'; 'U'
    def _get_slot_info(self, slot):
        return self.ran_config.slot_pattern[slot % self.ran_config.pattern_p] 
        

    def init(self):
        global UE_list
        self.slot = 0
        self.ul_uelist = UE_list 
        schedule_slot_info = self._get_slot_info(self.slot + PRE_SCHE_SLOT)
        initial_state = state(schedule_slot_info= schedule_slot_info, ul_uelist = self.ul_uelist, collision_list=[], success_list=[])
        return initial_state
    
    def reset(self):
        return self.init()
    

    def _reward(self, 
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
    


    ### Return 
    def send_DCI(self):
        ## Current slot calculate reward
        slot_info = self._get_slot_info(self.slot)
        reward = self._reward(slot_information = slot_info)

        ## Update state
        self.slot += 1
        schedule_slot_info = self._get_slot_info(self.slot + PRE_SCHE_SLOT)
        next_state = state(schedule_slot_info = schedule_slot_info, collision_list=[], success_list=[])

        if self.slot >= SIMULATION_FRAME * NUMBER_OF_SUBFRAME * pow(2, self.ran_config.numerology):
            self.done = True
        
        return next_state, reward, self.done
    
    def harq(self):
        ## Current slot calculate reward
        slot_info = self._get_slot_info(self.slot)
        reward = self._reward(slot_information = slot_info)

        ## Update state
        self.slot += 1
        schedule_slot_info = self._get_slot_info(self.slot + PRE_SCHE_SLOT)
        next_state = state(schedule_slot_info = schedule_slot_info, collision_list=[], success_list=[])

        if self.slot >= SIMULATION_FRAME * NUMBER_OF_SUBFRAME * pow(2, self.ran_config.numerology):
            self.done = True
        
        return next_state, reward, self.done
    

    def contenion(self, group_list : list):
        ## Current slot calculate reward
        success_ul_uelist = list()
        failed_ul_uelist = list()
        exp_RB, recv_RB = 0
        slot_info = self._get_slot_info(self.slot)

        for i in range(len(group_list)):
            group : Result = group_list[i]
            nrofRB = group.get_RB()
            ul_uelist = group.get_ul_uelist()
            rb_map = dict()
            collision_list, success_list = [None]*MAX_GROUP

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
                    for ue_id in range(len(rb_map[id])):
                        failed_ul_uelist.append(rb_map[id][ue_id])
                        collision_list[group.get_id()] += 1

                elif  len(rb_map[id]) == 1:
                    success_ul_uelist.append(rb_map[id][0])
                    success_list[group.get_id()] += 1
               
            # Calculate the reward
            success_ul_ue : UE
            for success_ul_ue in success_ul_uelist:
                recv_RB = recv_RB + success_ul_ue.nrofRB
            reward = self.reward(slot_information = slot_info, size_of_recv = recv_RB, size_of_exp=exp_RB, collision_list=failed_ul_uelist, success_list=success_ul_uelist)


            ## Update state
            self.slot += 1
            schedule_slot_info = self._get_slot_info(self.slot + PRE_SCHE_SLOT)
            next_state = state(schedule_slot_info = schedule_slot_info, collision_list = collision_list, success_list = success_list)

            if self.slot >= SIMULATION_FRAME * NUMBER_OF_SUBFRAME * pow(2, self.ran_config.numerology):
                self.done = True
            
            return next_state, reward, self.done
        
        
class Env:
    def __init__(self, args : argparse.Namespace):
        self.args = args
        self.ran_system = RAN_system(self.args)
            
    # return initial state
    def init(self):
        return self.ran_system.reset()

    def step(self, group_list : list):
        
        current_slot = self.ran_system.slot
        pattern_p = self.ran_system.ran_config.pattern_p
        slot_pattern = self.ran_system.ran_config.slot_pattern
        slot_info = slot_pattern[current_slot % pattern_p]

        next_state : state
        reward = 0
        done = False
        match slot_info:
            case 'D':
                next_state, reward, done = self.ran_system.send_DCI()
            case 'S':
                next_state, reward, done = self.ran_system.harq()
            case 'U':
                next_state, reward, done = self.ran_system.contenion(group_list = group_list)

        return next_state, reward, done

    def reset(self):
        return self.init()