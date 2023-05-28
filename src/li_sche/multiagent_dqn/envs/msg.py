import numpy as np

class MSG:
    def __init__(self):
        '''
        The payload begins with the UE_ID, the msg is identified with this.
        If the UE_ID = 0b0000000000000000 (0), it is the initial message.
        If the UE_ID = 0b1111111111111111 (65535), it is the ending message.
        Otherwise, represent the UE_ID (i.e., C_RNTI for UE)
        '''        
        self.UE_id : np.uint16 = 0b0000000000000000      
        self.payload : np.uint64 = 0


class SYNC(MSG):
    def __init__(self):
        super(SYNC, self).__init__()
        self.frame : np.uint16 = 0
        self.slot : np.uint16 = 0

    def fill_payload(self):
        '''
        Frame : 10 bits
        Slot : 4 bits
        '''
        size = 64
        pos = 0

        # Frame
        pos += 10
        self.payload |= (np.uint32(self.frame) & 0x3ff) << (size - pos)

        # Slot
        pos += 4
        self.payload |= (np.uint32(self.slot) & 0xf) << (size - pos)



# This msg is carried on PDCCH Physical Downlink Control Channel
class DCI_1_0(MSG):
    def __init__(self):
        super(DCI_1_0,self).__init__()
        self.format_indicator = 0                           # always 1 for DL
        self.frequency_domain_assignment = 0                # 
        self.time_domain_assignment = 0                     # 
        self.vrb_to_prb_mapping = 0                         # 0 : non-interleaved; 1 : interleaved
        self.mcs = 0                                        # Modulation and Coding Scheme 38.214 - 5.1.3.1-1/2
        self.ndi = 0                                        # 
        self.rv = 0                                         # 
        self.harq_process_number = 0                        #
        self.downlink_assignment_index = 0                  #
        self.tpc = 0                                        #
        self.pucch_resource = 0                             #
        self.pdsch_to_harq_feedback_timing_indicator = 0    # map to k1 = {1,2,3,4,5,6,7,8}

    def fill_payload(self):
        dci_size = 64
        pos = 0
        
        # fill DCI format               // 1 bit
        pos += 1
        self.payload |= (np.uint64(1)) << (dci_size - pos)

        # fill the frequency domain     // 16 bits (N_RB = 275 is the most)
        pos += 16
        self.payload |= (np.uint64(self.frequency_domain_assignment) & ((1 << 16) - 1)) << (dci_size - pos)

        # fill the time domain          // 4 bits
        pos += 4
        self.payload |= (np.uint32(self.time_domain_assignment) & 0xf) << (dci_size - pos)

        # VRB to PRB mapping            // 1 bit
        pos += 1
        self.payload |= (np.uint32(self.vrb_to_prb_mapping) & 1) << (dci_size - pos)

        # Modulation and Coding Scheme  // 5 bits
        pos += 5
        self.payload |= (np.uint32(self.mcs) & 0x1f) << (dci_size - pos)

        # New data indicator            // 1 bit
        pos += 1
        self.payload |= (np.uint32(self.ndi) & 1) << (dci_size - pos)

        # Redundancy version            // 2 bits
        pos +=2
        self.payload |= (np.uint32(self.rv) & 0x3) << (dci_size - pos)

        # HARQ process number           // 4 bits
        pos += 4
        self.payload |= (np.uint32(self.harq_process_number) & 0xf) << (dci_size - pos)

        # Downlink assignment index     // 2 bits
        pos += 2
        self.payload |= (np.uint32(self.downlink_assignment_index) & 0x3) << (dci_size - pos)

        # TPC for scheduling PUCCH       // 2 bits
        pos += 2
        self.payload |= (np.uint32(self.tpc) & 0xf) << (dci_size - pos)

        # PUCCH resource indicator      // 3 bits
        pos += 3
        self.payload |= (np.uint32(self.pucch_resource) & 0x7) << (dci_size - pos)

        # PDSCH-to-HARQ_feedback timing indicator // 3 bits
        pos += 3
        self.payload |= (np.uint32(self.pdsch_to_harq_feedback_timing_indicator) & 0x7) << (dci_size - pos)
        

        

class DCI_0_0(MSG):
    def __init__(self):
        super(DCI_1_0,self).__init__()
        self.format_indicator = 0                       # always 0 for UL
        self.frequency_domain_resource_assignment = 0   # number of RB
        self.time_domain_resource_assignment = 0        # 
        self.frequceny_hopping_flag = 0                 #
        self.mcs = 0                                    # 38.214 - 6.1.4
        self.ndi = 0                                    #
        self.rv = 0                                     # 0, 1, 2, 3
        self.harq_process_number = 0                    #
        self.TPC_command_for_schedule_pusch = 0         # 38.213 - table 7.1.1-1
        self.SUL_indicator = 0                          # 0 for SUL not configured; 1 for otherwise
        #  For this research, the following are added   #
        self.contention = 0                             # indicate the contention flag
        self.contention_size = 0                        # the number of contention RB


    def fill_payload(self):
        dci_size = 64
        pos = 0
        
        # fill DCI format               // 1 bit
        pos += 1
        self.payload |= (np.uint64(1)) << (dci_size - pos)

        # fill the frequency domain     // 16 bits (N_RB = 275 is the most)
        '''
        The original Freq. domain only occupies 4 bits
        But in this design, we suppose that all the freq. resource can be used as contention-based reosurce.
        For the scheduled UE (contention-free), the value of freq. value still occupies 4 bits only.
        '''
        pos += 16
        self.payload |= (np.uint64(self.frequency_domain_assignment) & ((1 << 16) - 1)) << (dci_size - pos)

        # fill the time domain          // 4 bits
        pos += 4
        self.payload |= (np.uint32(self.time_domain_assignment) & 0xf) << (dci_size - pos)

        # frequency hopping             // 1 bit
        pos += 1
        self.payload |= (np.uint32(self.frequceny_hopping_flag) & 1) << (dci_size - pos)

        # Modulation and Coding Scheme  // 5 bits
        pos += 5
        self.payload |= (np.uint32(self.mcs) & 0x1f) << (dci_size - pos)

        # New data indicator            // 1 bit
        pos += 1
        self.payload |= (np.uint32(self.ndi) & 1) << (dci_size - pos)

        # Redundancy version            // 2 bits
        pos +=2
        self.payload |= (np.uint32(self.rv) & 0x3) << (dci_size - pos)

        # HARQ process number           // 4 bits
        pos += 4
        self.payload |= (np.uint32(self.harq_process_number) & 0xf) << (dci_size - pos)

        # TPC for scheduling PUCCH       // 2 bits
        pos += 2
        self.payload |= (np.uint32(self.TPC_command_for_schedule_pusch) & 0x3) << (dci_size - pos)

        # SUL indicator                  // 1 bit
        pos += 1
        self.payload |= (np.uint32(self.SUL_indicator) & 1) << (dci_size - pos)

        # contention flag                // 1 bit
        pos += 1
        self.payload |= (np.uint32(self.contention) & 1) << (dci_size - pos)

        # contention size (up to 15 RBs) // 4 bits
        pos += 4
        self.payload |= (np.uint32(self.contention_size) & 1) << (dci_size - pos)

    def config_dci(self, dci : np.uint64):
        pass




# This msg is carried on PUCCH Physical Uplink Control Channel
# [ 3GPP TS 38.212 - 6.3.1.1 / 6.3.2.1]
class UCI(MSG):
    def __init__(self):
        super(UCI,self).__init__()
        self.ANK_NACK = 0
        self.SR = 0
        self.CSI = 0
