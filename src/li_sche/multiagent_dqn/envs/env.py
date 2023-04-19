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
MAX_UPLINK_GRANT    = 16
MAX_GROUP           = 4
PRE_SCHE_SLOT       = 6
SIZE_PER_RB         = 400


## The global variable to store all the UE info in system ##
UE_list = []

def decode_json(dct):
    return  UE(
                id              = 0, 
                sizeOfData      = dct[SIZE],
                delay_bound     = dct[DELAY],
                window          = dct[WINDOW],
                service         = dct[SERVICE],
                nr5QI           = dct[NR5QI],
                errorrate       = dct[ER],
                type            = dct[TYPE],
                group           = -1,
                rb_id           = -1,
                nrofRB          = -1,
                is_SR_sent      = False)

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


class State :
    def __init__(self, 
                 schedule_slot_info = 'D',
                 ul_uelist = []):
        
        self.schedule_slot_info = schedule_slot_info
        self.ul_uelist = ul_uelist

    def rm_ue(self, ue : UE):
        self.ul_uelist.remove(ue)

    def set_schedule_slot_info(self, schedule_slot_info):
        self.schedule_slot_info = schedule_slot_info

    def get_schedule_slot_info(self):
        return self.schedule_slot_info

    def reset(self):
        global UE_list
        self.ul_uelist = UE_list
        self.schedule_slot_info = 'D'
 

class RAN_system(RAN_config):
    def __init__(self, args : argparse.Namespace):
        
        self.slot = 0
        self.args = args
        self.done = False
        super(RAN_system, self).__init__(BW = self.args.bw,
                                        numerology = self.args.mu,
                                        nrofRB = self.args.rb,
                                        k0 = self.args.k0,
                                        k1 = self.args.k1,
                                        k2 = self.args.k2,)

        ## To add UE data to global variable UE_list
        global UE_list
        self.slot = 0
        UE_list = []
        cur_path = os.path.dirname(__file__)
        new_path = os.path.join(cur_path, '../../../data/uedata.json')
        with open(new_path, 'r') as ue_file:
            UE_list = json.load(ue_file,object_hook=decode_json) 
        for i in range(len(UE_list)):
            UE_list[i].id = i + 1
            
        self.temp_UE_list = UE_list.copy()
        self.ul_uelist = list()

    ### Return 'D'; 'S'; 'U'
    def _get_slot_info(self, slot):
        return self.slot_pattern[slot % self.pattern_p]


    def init(self):
        self.slot = 0
        schedule_slot_info = self._get_slot_info(self.slot + PRE_SCHE_SLOT)
        current_slot_info = self._get_slot_info(self.slot)

        if current_slot_info == 'U':
            for _ in range(MAX_UPLINK_GRANT):
                self.ul_uelist = self.temp_UE_list.pop(0)

        return State(schedule_slot_info= schedule_slot_info, ul_uelist = self.ul_uelist)
    
    def reset(self):
        return self.init()
    

    def _reward(self, 
               slot_information,
               collision_number_map : dict = dict(), 
               success_number_map : dict = dict(),
               expect_data_map : dict = dict(),
               success_data_map : dict = dict()
    ):
        
        reward = 0
        ## These weight size equal to MAX_GROUP 
        weight_recv_data = [1,1,1,1]
        weight_exp_data = [0.8,0.8,0.8,0.8]
        weight_col = [1,1,1,1]
        weight_suc = [1,1,1,1]

        if slot_information == 'D' or slot_information == 'S':
            return 0
        else :
            for group_id in collision_number_map:
                reward += weight_recv_data[group_id]*success_data_map[group_id] \
                            - weight_exp_data[group_id] * expect_data_map[group_id] \
                                + weight_suc[group_id] * success_number_map[group_id] \
                                    - weight_col[group_id] * collision_number_map[group_id]
        
        return reward
    


    ### Return reward
    def send_DCI(self, slot_info):
        ## Current slot calculate reward
        reward = self._reward(slot_information = slot_info)
        return reward
    
    def harq(self, slot_info):
        ## Current slot calculate reward
        reward = self._reward(slot_information = slot_info)
        return reward
    

    def contenion(self, action : list, slot_info):
        ### The highest level parameterm
        ul_uelist = action

        ####### !~ { The group_id is 0 ~ MAX_GROUP-1 } ~! ##########
        ## Arrange the grouping map which store the every UE which is allocated to the group
        ## group_map                : { group_id : [ue, ue, ue, ue, ...], ...} 
        group_map = dict()

        ## To store the number of collision in each group
        ## collision_number_map    : { group_id : number_of_collision, ...}
        collision_number_map = dict() 

        ## To store the number of success in each group
        ## collision_number_map    : { group_id : number_of_success, ...}
        success_number_map = dict()

        ## To store the total (expected) uplink transmission RBs
        ## expect_data_map          : { group_id : nrofRB, ...}
        expect_data_map = dict()

        ## To store the successful uplink transmission data size
        ## success_data_map         : { group_id : nrofRB, ...}
        success_data_map = dict()

        ## Store each ue into map
        for ue in ul_uelist:
            group_id = ue.group
            if group_id in group_map:
                group_map[group_id].append(ue)
            else:
                group_map[group_id] = [ue]
        
        ## In each group, the UE will randomly select a RB to transmit information
        for group_id in group_map:
            collision_number_map[group_id] = 0
            success_number_map[group_id] = 0
            expect_data_map[group_id] = 0
            success_data_map[group_id] = 0
            
            group = group_map[group_id]
            # Calculate the total aviavlible RBs in one group
            total_RB = 0
            for i in range(len(group)):
                total_RB += group[i].nrofRB
            expect_data_map[group_id] = total_RB * SIZE_PER_RB

            # Each UE ramdonly select
            for i in range(len(group)):
                if group_id == 0: # The scheduled UE has no need to contention
                    group[i].set_RB_ID(i + 1)
                else:
                    group[i].set_RB_ID(random.randrange(total_RB) + 1)
            
            # Perform contention in every group
            # rb_map : {rb_id : [ue, ue, ...], ...}
            rb_map = dict()
            for ue in group:
                rb_id = ue.rb_id
                if rb_id in rb_map:
                    rb_map[rb_id].append(ue)
                else:
                    rb_map[rb_id] = [ue]

            # To update the collision and success map
            # update the self.ul_uelist if the ul_ue success to transmit the data
            for rb_id in rb_map:

                # Contention failed
                if len(rb_map[rb_id]) > 1:
                    collision_number_map[group_id] += len(rb_map[rb_id])

                # Contention resolution
                else:
                    success_number_map[group_id] += len(rb_map[rb_id])
                    ue_id = rb_map[rb_id][0].id
                    size = rb_map[rb_id][0].nrofRB * SIZE_PER_RB
                    for i in range(len(self.ul_uelist)):                        
                        if ue_id == self.ul_uelist[i].id:
                            success_data_map[group_id] += size
                            self.ul_uelist[i].decay_size(size)
                            

        ## Out of the group loop
        ## Calculate the reward
        reward = self._reward(slot_information = slot_info, \
                                collision_number_map = collision_number_map, \
                                    success_number_map=success_number_map, \
                                        expect_data_map=expect_data_map, \
                                            success_data_map=success_data_map)

        return reward

    def step(self, action:list):
        # To calculate the reward
            
        current_slot_info = self._get_slot_info(self.slot)
        reward = 0
        match current_slot_info:
            case 'D':
                reward = self.send_DCI(slot_info = 'D')
            case 'S':
                reward = self.harq(slot_info = 'S')
            case 'U':
                reward = self.contenion(action = action, slot_info = 'U')


        ## Update state
        self.slot += 1
        schedule_slot_info = self._get_slot_info(self.slot + PRE_SCHE_SLOT)
        current_slot_info = self._get_slot_info(self.slot)

        ## Update the UEs which have been already scheduled
        for i in range(len(self.ul_uelist)):
            self.ul_uelist[i].decay_delay(1)
        ue : UE
        for ue in self.ul_uelist:
            if ue.sizeOfData <= 0 or ue.delay_bound <= 0:
                self.ul_uelist.remove(ue)
        
        ## Add other UEs which have not been scheduled if there are some.
        ## Only in UL slot, the UE can send Scheduling Request
        if current_slot_info == 'U':
            for _ in range(MAX_UPLINK_GRANT):
                if len(self.temp_UE_list):
                    self.ul_uelist.append(self.temp_UE_list.pop(0))
                else:
                    break


        if self.slot >= SIMULATION_FRAME * NUMBER_OF_SUBFRAME * pow(2, self.numerology):
            self.done = True
        if len(self.temp_UE_list) == 0 and len(self.ul_uelist) == 0:
            self.done = True

        next_state = State(schedule_slot_info = schedule_slot_info, ul_uelist = self.ul_uelist)

        return next_state, reward, self.done
        
        
class Env:
    def __init__(self, args : argparse.Namespace):
        self.args = args
        self.ran_system = RAN_system(self.args)
        self.action_map = dict()
            
    # return initial state
    def init(self):
        return self.ran_system.reset()

    ## Action is store in ul_uelist
    def step(self, action : list):
        schedule_slot = (self.ran_system.slot + PRE_SCHE_SLOT)
        self.action_map[schedule_slot] = action.copy()
        
        action_ul_list = list()
        if self.ran_system.slot in self.action_map:
            action_ul_list = self.action_map[self.ran_system.slot].copy()
            del self.action_map[self.ran_system.slot]

        next_state, reward, done = self.ran_system.step(action = action_ul_list)
        return next_state, reward, done

    def reset(self):
        return self.init()