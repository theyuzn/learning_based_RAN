'''
Creater Chuang, Yu-Hsin
Lab : MWNL -- Mobile & Wireless Networking Labtory
Advisor : S.T. Sheu
Copyright @Brandon, @Yu-Hsin Chuang, @Chuang, Yu-Hsin.
All rights reserved.

Created in 2023/03
'''

import argparse
import math
import random
import numpy as np
import json
import os

import li_sche.utils.pysctp.sctp as sctp

from collections import namedtuple, deque
from .ue import UE
from .thread import Socket_Thread
from .msg import *
import  li_sche.multiagent_dqn.envs.msg as MSG_HDR


## Constant
# k0 = 0, k1 = 0 ~ 2, k2 = 3
SLOT_PATTERN1       = ['D','D','S','U','U']
PATTERN_P1          = 5

# k0 = 0, k1 = 0 ~ 5, k2 = 4 ~ 6
SLOT_PATTERN2       = ['D','D','D','D','D','S','U', 'U','U','U']
PATTERN_P2          = 10

SIMULATION_FRAME    = 200
NUMBER_OF_SUBFRAME  = 10
MAX_UPLINK_GRANT    = 16
MAX_GROUP           = 4
PRE_SCHE_SLOT       = 6
SIZE_PER_RB         = 400
LEGACY_CONSTANT     = 120

class RAN:
    def __init__(self, 
                 BW         = 1600, #MHz 
                 numerology = 5, 
                 nrofRB     = 248, 
                 k0         = 0,
                 k1         = 0,
                 k2         = 3,
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
        self.spf            = 10 * 2 * pow(2, numerology)
        self.tbs_per_RB     = 848 # bits For MCS = 27, Layer = 1, each RB

class Schedule_Result:
    '''
    For now, only has two results
    1.) DCI msgs --> dci0_0 and dci1_0
    2.) USCH shceduling results
    '''
    def __init__(self):
        self.DCI_Transition = namedtuple('DCI_Tuple', ('dci0', 'dci1'))
        self.USCH_Transition = namedtuple('Data_Tuple', ('frame', 'slot', 'nrof_UE', 'cumulatied_rb'))

        self.DCCH = self.DCI_Transition(dci0=list(), dci1 = list())          
        self.USCH = self.USCH_Transition(frame=0, slot=0, nrof_UE=0, cumulatied_rb=0)    

    def get_DCCH(self):
        return self.DCI_Transition(self.DCCH)

    def get_USCH(self):
        return self.USCH_Transition(self.USCH)  


class RAN_system(RAN):
    '''
    ### Purpose
    This work is providing the high throughput 5G scheduler in eMBB system within low delay.
    Without considering the 1.Fairness, 2.Channel Condition.

    ### RAN system
    This is palying a role as the state in DRL.
    Input is alway the UE's request (i.e., Uplink msg)
    Ex. : UCI (i.e., scheduling request / Special slot) and Data (i.e., UL data + BSR / UL slot)

    
    The scheduling result is sent to UE through DCI msg in DL Slot.

    * Msg transmission between UE entity and gNB entity is implemented by the SCTP socket programming.
    * The DCI is sent from gNB to UE (over PDCCU) in DL slot or Special slot
    * The UCL is sent from UE to gNB (over PUCCH) in Special slot
    * The msg is sent from UE to gNB (over PUSCH) in UL slot
    * The bsr is send from UE to gNB (over PUSCH) in UL slot
    The (physical channel) is not implemented yet. maybe no need to implement.

    ### TODO ... 
    --> In each phase, the reward need to be re-designed.
    --> In Reforcement learning, you need to design the reward function to achieve the goal you want.
    1. Take Fairness, Channel Condition into account.
    2. Consider other types of services in 5G and beyond
        Ex. URLLC (without configured grant), mMTC, ...
    3. The DL schedule algorithm
    4. The DRL in the UE side (The UE need to learn in the MAC layer)
    5. The Federated Learning in UE side.
    6. Cooperative multi-agent in the both side (UE and gNB)
    '''

    def __init__(self, args : argparse.Namespace):
        super(RAN_system, self).__init__(BW = args.bw,
                                        numerology = args.mu,
                                        nrofRB = args.rb)
        self.State_Transition = namedtuple('State_Tuple', ('frame', 'slot', 'ul_req'))
        self.frame = 0
        self.slot = 0
        self.args = args
        self.done = False

        self.UE_List = deque([], maxlen = 65535)
        self.UL_Flow_UE = deque([], maxlen = 65535)

        self.USCH_ra_queue =  deque([], maxlen = 5) # ULSCH resource allocation; depends on the number of UL slot in a aperiod
        self.i = 0

    def decode_json(self, dct):
        self.i += 1
        return  UE(
                    id              = self.i, 
                    bsr             = 1, # math.ceil(dct['sizebyte']/self.tbs_per_RB),
                    rdb             = dct['delayms']*math.pow(2, 1),
                    )

    def init_RAN_system(self):
        # Initial the system
        self.frame = 0
        self.slot = 0
        self.done = False
        reward = 0

         # Load the UEs
        path = f"{os.path.dirname(__file__)}/../../../data/uedata.json"
        with open(path, 'r') as ue_file:
            self.UE_List = deque(json.load(ue_file,object_hook=self.decode_json), maxlen = 65535)
        state_tuple = self.State_Transition(frame = self.frame, slot = self.slot, ul_req = [])
        return state_tuple, reward, self.done

   
    # API for every entity
    def get_slot_info(self, frame, slot):
        # return 'D'; 'S'; 'U'
        cumulated_slot = frame*self.spf + slot
        return self.slot_pattern[cumulated_slot % self.pattern_p]


    def contenion(self, nrof_UE = 0, cumulated_rb = 0, ul_data = deque([], maxlen = 65535)):
        reward = 0
        total_rb_list = [list()]*self.nrofRB

        len_ue = len(ul_data)
        for _ in range(len_ue):
            ue : UE = ul_data.popleft()
            
            for j in range(ue.freq_len):
                total_rb_list[ue.start_rb + j].append(ue)

        succ_list = list()
        fail_list = list()
        succ_rb = 0
        fail_rb = 0
        contention_ue_list : list
        for contention_ue_list in total_rb_list:
            if len(contention_ue_list) == 0:
                continue
            elif len(contention_ue_list) == 1:
                succ_list.append(contention_ue_list.pop())
                succ_rb += 1
            else:
                for fail_ue in contention_ue_list:
                    fail_list.append(fail_ue)
                    fail_rb += 1
        
        reward = succ_rb - fail_rb - LEGACY_CONSTANT        

        return reward, succ_list, fail_list

    
    def step(self, action : Schedule_Result):
        # Initial
        ul_req = deque([], maxlen = 65535)

        # Tx / Rx 
        current_slot_info = self.get_slot_info(self.frame, self.slot)

        reward = 0
        match current_slot_info:
            case 'D':
                DCI0 = deque(action.DCCH.dci0, maxlen = 65535)
                for dci_bytes_payload in DCI0:
                    msg = MSG()
                    msg.payload = dci_bytes_payload
                    header = msg.decode_header()

                    dci0_0 : DCI_0_0 = DCI_0_0()
                    dci0_0.header = header
                    dci0_0.payload = msg.payload
                    dci0_0.decode_msg()

                    for i in range(len(self.UL_Flow_UE)):
                        if dci0_0.UE_id == self.UL_Flow_UE[i].id:
                            self.UL_Flow_UE[i].start_rb = dci0_0.start_rb
                            self.UL_Flow_UE[i].freq_len = dci0_0.freq_len
                            self.UL_Flow_UE[i].transmission_time = (self.frame * self.spf) + self.slot + self.k2

                            if dci0_0.contention:
                                self.UL_Flow_UE[i].contention = True
                                self.UL_Flow_UE[i].start_rb = dci0_0.start_rb + random.randrange(dci0_0.contention_size - 1)
                            
                            break

                self.USCH_ra_queue.append(action.USCH)


            case 'S':
                for i in range(16):
                    if len(self.UE_List) > 0:
                        ul_ue = self.UE_List.popleft()
                        self.UL_Flow_UE.append(ul_ue)
                        ul_req.append(ul_ue)
                
            case 'U':
                if len(self.USCH_ra_queue) > 0:

                    USCH_ra = self.USCH_ra_queue.popleft()
                    if USCH_ra.frame != self.frame or USCH_ra.slot != self.slot:
                        reward = -1
                    else:
                        nrof_UE = USCH_ra.nrof_UE
                        cumulated_rb = USCH_ra.cumulatied_rb

                        if nrof_UE > 0:
                            print("Perform contention")
                            contention_ue = deque([], maxlen = 65535)
                            queuing_ue = deque([], maxlen = 65535)

                            for i in range(len(self.UL_Flow_UE)):
                                q_ue : UE = self.UL_Flow_UE.popleft()
                                if q_ue.transmission_time == (self.frame * self.spf) + self.slot:
                                    contention_ue.append(q_ue)
                                else:
                                    q_ue.queuing_delay += self.pattern_p
                                    queuing_ue.append(q_ue)

                            reward , suc_ue, fail_ue = self.contenion(nrof_UE=nrof_UE, cumulated_rb=cumulated_rb, ul_data=contention_ue)

                            ue : UE
                            self.UL_Flow_UE = queuing_ue.copy()
                            for ue in suc_ue:
                                ue.send_cnt += 1
                                ue.suc_cnt += 1
                                ul_req.append(ue)
                            for ue in fail_ue:
                                ue.queuing_delay += self.pattern_p
                                ue.fail_cnt += 1
                                self.UL_Flow_UE.append(ue)
                            
                
        
        # Update the slot
        self.slot += 1
        if self.slot >= self.spf:
            self.slot = 0
            self.frame += 1
            
        if self.frame >= SIMULATION_FRAME:
            print(self.done)
            self.done = True
        
        # Inform the UE entity
        if self.done:
            next_state_tuple = self.State_Transition(frame = self.frame, slot = self.slot, ul_req = [])

        next_state_tuple = self.State_Transition(frame = self.frame, slot = self.slot, ul_req = ul_req)
        return next_state_tuple, reward, self.done