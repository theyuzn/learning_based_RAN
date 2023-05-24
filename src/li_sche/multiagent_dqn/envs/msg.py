import numpy as np

class MSG:
    def __init__(self):
        self.UE_id : np.uint16          # To represent the RNTI of UE 1 ~ 65535
        self.dci_0_0 : DCI_0_0 = None
        self.dci_1_0 : DCI_1_0 = None
        self.uci : UCI = None

        # [ the bit string] #
        # UE ID for 8 bits, 
        # If the UE ID = 0, it is initial msg.
        # If the UE_ID = 0xff, it is the ending msg
        self.payload = 0

    def encode(self):

        if self.dci_0_0 != None:
            self.payload = 0b0
            

            
            return self.payload
        
        if self.dci_1_0 != None:
            return self.payload
        
        if self.uci != None:
            return self.payload
        
    def decode_header(self, msg : np.int64 = 0x00000000):
        id_mask = 0b11111111
        id = id_mask & msg
        id = id >>24

        if id == 0x00:
            return "Initial"
        elif id == 0xff:
            return "End"
        else:
            return "MSG"
        
    

# This msg is carried on PDCCH Physical Downlink Control Channel
class DCI_1_0(MSG):
    def __init__(self):
        super(DCI_1_0,self).__init__()
        self.format_indicator = 0                           # [1] always 1 for DL
        self.frequency_domain_assignment = 0                # [variable] 1 ~ 16
        self.time_domain_assignment = 0                     # [4]
        self.vrb_to_prb_mapping = 0                         # [1] 0 for non-interleaved; 1 for otherwise
        self.mcs = 0                                        # [5] Modulation and Coding Scheme 38.214 - 5.1.3.1-1/2
        self.ndi = 0                                        # [1] New Data Indicator 
        self.rv = 0                                         # [2] Redundency Version 
        self.harq_process_number = 0                        # [4]
        self.downlink_assignment_index = 0                  # [2]
        self.tpc = 0                                        # [2]
        self.pucch_resource = 0                             # [3]
        self.pdsch_to_harq_feedback_timing_indicator = 0    # [3] map to k1 = {1,2,3,4,5,6,7,8}
        

class DCI_0_0(MSG):
    def __init__(self):
        super(DCI_1_0,self).__init__()
        self.format_indicator = 0                       # [1] always 0 for UL
        self.frequency_domain_resource_assignment = 0   # [4] number of RB
        self.time_domain_resource_assignment = 0        # [variable]
        self.frequceny_hopping_flag = 0                 # [1]
        self.mcs = 0                                    # [5] 38.214 - 6.1.4
        self.ndi = 0                                    # [1]
        self.rv = 0                                     # [2] 0, 1, 2, 3
        self.harq_process_number = 0                    # [4]
        self.TPC_command_for_schedule_pusch = 0         # [2] 38.213 - table 7.1.1-1
        self.SUL_indicator = 0                          # [1] 0 for SUL not configured; 1 for otherwise
        #  For this research, the following are added   #
        self.contention = 0                             # [1] # indicate the contention flag
        self.contention_size = 0                        # [4] # the number of contention RB


# This msg is carried on PUCCH Physical Uplink Control Channel
# [ 3GPP TS 38.212 - 6.3.1.1 / 6.3.2.1]
class UCI(MSG):
    def __init__(self):
        super(UCI,self).__init__()
        self.ANK_NACK = 0
        self.SR = 0
        self.CSI = 0
