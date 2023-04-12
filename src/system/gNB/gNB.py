import argparse

from ue.ue import UE
from scheduler.lightweight_scheduler.dqn import DQNAgent 

simulation_frame    = 200 # 20 seconds
nrofRB              = 106 # BW = 40 MHz ; SCS = 30kHz (for now)
mu                  = 2
nrofsubframe        = 10
nrf_slot            = simulation_frame*nrofsubframe*pow(2,mu)
pre_sche            = 6 # schedule the slot after 6 slots from present
slot_pattern        = ['D','D','S','U','U']
patter_p            = 5
k0                  = 0
k1                  = 2
k2                  = 4

class gNB :
    def __init__(self, parser : argparse.Namespace , uelist = []):
        self.uelist = uelist
        self.parser = parser
        self.dqnAgent = DQNAgent(args = parser)

        if parser.load_latest and not parser.checkpoint:
            self.dqnAgent.load_latest_checkpoint()
        elif parser.checkpoint:
            self.dqnAgent.load_checkpoint(parser.checkpoint)

        return
    

    def start(self):

        slot = 0
        while True:
            slot_type = slot_pattern[slot%patter_p]

            # self.dqnAgent.train()

            #####################################################################################
            # For the DL slot                                                                   #
            #       Send DCI (NDI + UL grant + Resource)                                        #
            #                                                                                   #
            # For the SL                                                                        #
            #       Receive HARQ, no use for now due to the assumption that channel is perfect  #
            #                                                                                   #
            # For the UL slot                                                                   #
            #       To receive UL data and watch for the situation of contention                #
            #                                                                                   #
            #####################################################################################
            match slot_type:
                case 'D':
                   self.dqnAgent.DL() 
                case 'S':
                    self.dqnAgent.SL() 
                    # Do UCI            -- for HARQ
                    # For now is no use -- the channel is perfect
                    break
                case 'U':
                    self.dqnAgent.UL() 
                    break

            slot = slot + 1
