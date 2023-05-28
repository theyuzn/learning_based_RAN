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
from .ue import UE
import random
import li_sche.utils.pysctp.sctp as sctp
from .msg import MSG

from .thread import Socket_Thread

## Constant
# k0 = 0, k1 = 0 ~ 2, k2 = 2 ~ 3
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


INITIAL = 0x00
END = 0xff

class RAN:
    def __init__(self, 
                 BW         = 40, 
                 numerology = 1, 
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


class State :
    def __init__(self, 
                 current_slot_info = 'D',
                 schedule_slot_info = 'D',
                 ul_req = list()):
        self.current_slot_info = current_slot_info
        self.schedule_slot_info = schedule_slot_info
        self.ul_req = ul_req

    def reset(self):
        self.current_slot_info = 'D'
        self.schedule_slot_info = 'D'
        self.ul_req = list()

    def rm_ue(self, ue : UE):
        self.ul_req.remove(ue)
 

class RAN_system(RAN):
    '''
    ### Purpose
    This work is providing the high throughput 5G scheduler in eMBB system within low delay.
    Without considering the 1.Fairness, 2.Channel Condition.

    ### RAN system
    Input is alway the UE's request (i.e., Uplink msg)
    Ex. : UCI (i.e., scheduling request / Special slot) and Data (i.e., UL data + BSR / UL slot)

    Output is the State@class which contains the system information
    The State is consist of Current slot + Sche slot + UL_req
    For agent to train or inference the scheduler algorithm
    
    The scheduling result is sent to UE through DCI msg in DL Slot.

    * How to implement the msg transmission between UE entity and gNB entity ?
        Use SCTP socket to send the msg.
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
    def __init__(self, args : argparse.Namespace, conn_sock : sctp.sctpsocket_tcp):
        super(RAN_system, self).__init__(BW = args.bw,
                                        numerology = args.mu,
                                        nrofRB = args.rb,
                                        k0 = args.k0,
                                        k1 = args.k1,
                                        k2 = args.k2)
        self.slot = 0
        self.args = args
        self.done = False
        self.ul_req = list()
        self.conn_sock = conn_sock
        while True:
            self.conn_sock.sctp_send("tset")

    def callback(self, msg : bytes):
        print(msg)


    ### Return 'D'; 'S'; 'U'
    def _get_slot_info(self, slot):
        return self.slot_pattern[slot % self.pattern_p]
    
    def gNB_send(self, msg : MSG):
        print(msg)
        print(type(msg))
        self.conn_sock.sctp_send(msg)

    def init(self):
        self.slot = 0
        schedule_slot_info = self._get_slot_info(self.slot + PRE_SCHE_SLOT)
        current_slot_info = self._get_slot_info(self.slot)
        self.ul_req = list()
        self.done = False
        self.gNB_send(INITIAL.to_bytes(2,'big'))

        return State(current_slot_info = current_slot_info,  
                     schedule_slot_info= schedule_slot_info, 
                     ul_req = self.ul_req)
    
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


        if self.slot >= SIMULATION_FRAME * NUMBER_OF_SUBFRAME * math.pow(2, self.numerology):
            self.done = True
        if len(self.temp_UE_list) == 0 and len(self.ul_uelist) == 0:
            self.done = True

        next_state = State(schedule_slot_info = schedule_slot_info, ul_uelist = self.ul_uelist)

        return next_state, reward, self.done