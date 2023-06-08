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
import li_sche.utils.pysctp.sctp as sctp

from collections import namedtuple
from .ue import UE
from .thread import Socket_Thread
from .msg import *



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

class RAN:
    def __init__(self, 
                 BW         = 1600, #MHz 
                 numerology = 5, 
                 nrofRB     = 248, 
                 k0         = 0,
                 k1         = 1,
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

class Schedule_Result:
    def __init__(self):
        self.DCCH = None    # Send the DCI0_0 msg to each UE (Does not send DCI#1_0)
        self.DSCH = None    # Send DL data (Skip for now...)
        self.UCCH = None    # UL resources for receiving the UCI
        self.USCH = None    # UL resources for receiving the UL data from UEs


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
    def __init__(self, args : argparse.Namespace, send_sock : sctp.sctpsocket_tcp, recv_sock : sctp.sctpsocket_tcp):
        super(RAN_system, self).__init__(BW = args.bw,
                                        numerology = args.mu,
                                        nrofRB = args.rb)
        self.State_Transition = namedtuple('State_Tuple', ('frame', 'slot', 'ul_req'))
        self.recv_sock = recv_sock
        self.send_sock = send_sock
        self.frame = 0
        self.slot = 0
        self.args = args
        self.done = False
        self.ul_req = list()

    def init_RAN_system(self):
        # Inform UE entity
        init_msg = INIT()
        init_msg.header = HDR_INIT
        init_msg.k0 = self.k0
        init_msg.k2 = self.k2
        init_msg.fill_payload()
        self.downlink_channel(init_msg.payload)

        # Initial the system
        self.frame = 0
        self.slot = 0
        self.done = False
        self.ul_req = list()
        reward = 0
        state_tuple = self.State_Transition(frame = self.frame, slot = self.slot, ul_req = self.ul_req)
        return state_tuple, reward, self.done

   
    # API for every entity
    def get_slot_info(self, frame, slot):
        # return 'D'; 'S'; 'U'
        cumulated_slot = frame*self.spf + slot
        return self.slot_pattern[cumulated_slot % self.pattern_p]
    
    def uplink_channel(self):
        ul_end = False
        msg_list = list()
        while not ul_end:
            fromaddr, flags, msg, notif = self.recv_sock.sctp_recv(65535)
            msg = int.from_bytes(msg, "big")
            msg_list.append(msg)
        return msg_list
        
    def downlink_channel(self, msg : int):
        print(f"[gNB] Sned {bin(msg)}")
        msg = msg.to_bytes(16, "big")
        self.send_sock.sctp_send(msg)


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

        for i in range(MAX_GROUP):
            group_map[i]            = []
            collision_number_map[i] = 0
            success_number_map[i]   = 0
            expect_data_map[i]      = 0
            success_data_map[i]     = 0

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

    def send_DCI(self):
        pass

    def recv_UCI(self):
        uci_packets = list()
        uci_packets = self.uplink_channel()
        

    def recv_Data(self):
        ul_packets = list()
        ul_packets = self.uplink_channel()
        
    

    def step(self, schedule : Schedule_Result):
        # Update the slot
        self.slot += 1
        if self.slot >= self.spf:
            self.slot = 0
            self.frame += 1

        if self.frame >= SIMULATION_FRAME:
            self.done = True
        
        # Inform the UE entity
        if self.done:
            end_msg = MSG()
            end_msg.header = HDR_END
            end_msg.fill_payload()
            self.downlink_channel(end_msg.payload)
            next_state_tuple = self.State_Transition(frame = self.frame, slot = self.slot, ul_req = [])
            return next_state_tuple, 0, self.done

        
        # Slot indication to UE entity
        slot_ind = SYNC()
        slot_ind.frame = self.frame
        slot_ind.slot = self.slot
        slot_ind.fill_payload()
        self.downlink_channel(slot_ind.payload)

        # Tx / Rx 
        current_slot_info = self.get_slot_info(self.frame, self.slot)
        

        reward = 0
        match current_slot_info:
            case 'D':
                # reward = self.send_DCI(slot_info = 'D')
                pass
            case 'S':
                # reward = self.harq(slot_info = 'S')
                pass
            case 'U':
                # reward = self.contenion(action = action, slot_info = 'U')
                pass

        

        
        next_state_tuple = self.State_Transition(frame = self.frame, slot = self.slot, ul_req = [])
        return next_state_tuple, reward, self.done