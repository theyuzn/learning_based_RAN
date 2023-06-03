import numpy as np

class MSG:
    def __init__(self):
        '''
        1.) DownLink Message
        The payload begins with the header, the msg is identified with this.
        If the header = 0b0000000000000000 (0), it is the initial message.
        If the header = 0b1111111111111111 (65535), it is the ending message.
        If the header = 0b1111111111111110 (65534), it is the sync message.
        Otherwise, represent the UE_id (i.e., C_RNTI for UE)

        2.) Uplink Message
        The UCI is focusing on the Scheduling Request (SR) after the header
        '''        
        self.header : np.uint16 = 0b0000000000000000 
        self.type = ''  # {'Init', 'End', 'Sync', 'Msg'}
        self.payload : np.uint64 = 0

    def decode_header(self):
        size = 64
        pos = 0

        # unpack the header          // 16 bits
        pos += 16
        header = (self.payload >> (size - pos)) & ((1 << 16) - 1)

        if header == 0b0000000000000000 :
            self.type = 'Init'
        elif header == 0b1111111111111111:
            self.type = 'End'
        elif header == 0b1111111111111110:
            self.type = 'Sync'
        else:
            self.type = 'Msg'
        
        return self.type
            

class SYNC(MSG):
    def __init__(self):
        super(SYNC, self).__init__()
        self.frame : np.uint16 = 0
        self.slot : np.uint16 = 0

    def fill_payload(self):
        '''
        header  : 16 bits
        Frame   : 10 bits
        Slot    : 4 bits
        '''
        self.header = 0b1111111111111110
        size = 64
        pos = 0
        
        # fill UE ID                    // 16 bits
        pos += 16
        self.payload |= (np.uint64(self.header) & ((1 << 16) - 1)) << (size - pos)

        # Frame
        pos += 10
        self.payload |= (np.uint32(self.frame) & 0x3ff) << (size - pos)

        # Slot
        pos += 4
        self.payload |= (np.uint32(self.slot) & 0xf) << (size - pos)

    def decode_msg(self):
        size = 64
        pos = 16                # skip the header // 16 bits

        # unpack frame          // 10 bits
        pos += 10
        self.frame = (self.payload >> (size - pos)) & 0x3ff

        # unpack slot           // 4 bits
        pos += 4
        self.slot = (self.payload >> (size - pos)) & 0xf

class DCI(MSG):
    def __init__(self):
        super(DCI, self).__init__()
        self.DCI_format = 0 # 0 : UL DCI, 1 : DL DCI

    def decode_DCI_foramt(self, msg : np.uint64):
        dci_size = 64
        pos = 16 # skip the header
        
        # unpack the DCI foramt
        pos += 1
        self.DCI_format = (self.payload >> (dci_size - pos)) & 1

        return self.DCI_format


# This msg is carried on PDCCH Physical Downlink Control Channel
class DCI_1_0(DCI):
    def __init__(self):
        super(DCI_1_0,self).__init__()
        self.UE_id = 0                                      # The UE ID for 16 bits
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

        # fill UE ID                    // 16 bits
        pos += 16
        self.payload |= (np.uint64(self.header) & ((1 << 16) - 1)) << (dci_size - pos)

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
        self.payload |= (np.uint32(self.tpc) & 0x3) << (dci_size - pos)

        # PUCCH resource indicator      // 3 bits
        pos += 3
        self.payload |= (np.uint32(self.pucch_resource) & 0x7) << (dci_size - pos)

        # PDSCH-to-HARQ_feedback timing indicator // 3 bits
        pos += 3
        self.payload |= (np.uint32(self.pdsch_to_harq_feedback_timing_indicator) & 0x7) << (dci_size - pos)

    def decode_msg(self, msg : np.uint64):
        pos = 0
        dci_size = 64
        self.payload = msg

        # unpack the header          // 16 bits
        pos += 16
        self.header = (self.payload >> (dci_size - pos)) & ((1 << 16) - 1)
        self.UE_id = self.header

        # unpack DCI format               // 1 bit
        pos += 1
        self.format_indicator = (self.payload >> (dci_size - pos)) & 1

        # unpack the frequency domain     // 16 bits (N_RB = 275 is the most)
        pos += 16
        self.frequency_domain_assignment = (self.payload >> (dci_size - pos)) & ((1 << 16) - 1)

        # unpack the time domain          // 4 bits
        pos += 4
        self.time_domain_assignment = (self.payload >> (dci_size - pos)) & 0xf

        # VRB to PRB mapping            // 1 bit
        pos += 1
        self.vrb_to_prb_mapping = (self.payload >> (dci_size - pos)) & 1

        # Modulation and Coding Scheme  // 5 bits
        pos += 5
        self.mcs = (self.payload >> (dci_size - pos)) & 0x1f

        # New data indicator            // 1 bit
        pos += 1
        self.ndi = (self.payload >> (dci_size - pos)) & 1

        # Redundancy version            // 2 bits
        pos +=2
        self.rv = (self.payload >> (dci_size - pos)) & 0x3

        # HARQ process number           // 4 bits
        pos += 4
        self.harq_process_number = (self.payload >> (dci_size - pos)) & 0xf

        # Downlink assignment index     // 2 bits
        pos += 2
        self.downlink_assignment_index = (self.payload >> (dci_size - pos)) & 0x3

        # TPC for scheduling PUCCH       // 2 bits
        pos += 2
        self.tpc = (self.payload >> (dci_size - pos)) & 0x3

        # PUCCH resource indicator      // 3 bits
        pos += 3
        self.pucch_resource = (self.payload >> (dci_size - pos)) & 0x7

        # PDSCH-to-HARQ_feedback timing indicator // 3 bits
        pos += 3
        self.pdsch_to_harq_feedback_timing_indicator = (self.payload >> (dci_size - pos)) & 0x7
        

class DCI_0_0(DCI):
    def __init__(self):
        super(DCI_1_0,self).__init__()
        self.UE_id = 0                                  # The 16 bits UE id
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

        # fill header                    // 16 bits
        pos += 16
        self.payload |= (np.uint64(self.header) & ((1 << 16) - 1)) << (dci_size - pos)
        
        # fill DCI format               // 1 bit
        pos += 1
        self.payload |= (np.uint64(1)) << (dci_size - pos)

        # fill the frequency domain     // 4 bits
        pos += 4
        self.payload |= (np.uint64(self.frequency_domain_assignment) & 0xf) << (dci_size - pos)

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

        # contention size (up to 275 RBs, all RB are arranged to be contention) // 16 bits
        '''
        This field indicates the range of contention-based RB size.
        '''
        pos += 16
        self.payload |= (np.uint32(self.contention_size) & ((1 << 16) - 1)) << (dci_size - pos)

    def decode_msg(self, msg : np.uint64):
        pos = 0
        dci_size = 64
        self.payload = msg

        # unpack the header          // 16 bits
        pos += 16
        self.header = (self.payload >> (dci_size - pos)) & ((1 << 16) - 1)
        self.UE_id = self.header

        # unpack DCI format               // 1 bit
        pos += 1
        self.format_indicator = (self.payload >> (dci_size - pos)) & 1

        # unpack the frequency domain     // 4 bits
        pos += 4
        self.frequency_domain_assignment = (self.payload >> (dci_size - pos)) & 0xf

        # unpack the time domain          // 4 bits
        pos += 4
        self.time_domain_assignment = (self.payload >> (dci_size - pos)) & 0xf

        # frequency hopping               // 1 bit
        pos += 1
        self.frequceny_hopping_flag = (self.payload >> (dci_size - pos)) & 1

        # Modulation and Coding Scheme  // 5 bits
        pos += 5
        self.mcs = (self.payload >> (dci_size - pos)) & 0x1f

        # New data indicator            // 1 bit
        pos += 1
        self.ndi = (self.payload >> (dci_size - pos)) & 1

        # Redundancy version            // 2 bits
        pos +=2
        self.rv = (self.payload >> (dci_size - pos)) & 0x3

        # HARQ process number           // 4 bits
        pos += 4
        self.harq_process_number = (self.payload >> (dci_size - pos)) & 0xf

        # TPC for scheduling PUCCH       // 2 bits
        pos += 2
        self.TPC_command_for_schedule_pusch = (self.payload >> (dci_size - pos)) & 0x3

        # unpack the SUL indicator      // 1 bit
        pos += 1
        self.SUL_indicator = (self.payload >> (dci_size - pos)) & 1

        # unpack contention flag     // 1 bits
        pos += 1
        self.contention = (self.payload >> (dci_size - pos)) & 1

        # unpack contention size    // 16 bits
        pos += 16
        self.contention_size = (self.payload >> (dci_size - pos)) & ((1 << 16) - 1)


        




# This msg is carried on PUCCH Physical Uplink Control Channel
# [ 3GPP TS 38.212 - 6.3.1.1 / 6.3.2.1]
class UCI(MSG):
    def __init__(self):
        super(UCI,self).__init__()
        self.ACK_NACK = 0
        self.proc_id = 0
        self.SR = 0
        self.CSI = 0

    def fill_payload(self):
        uci_size = 64
        pos = 0

        # fill the header           // 16 bits
        pos += 16
        self.payload |= (np.uint64(self.header) & ((1 << 16) - 1)) << (uci_size - pos)

        # fill the porc_id          // 4 bits
        pos += 4
        self.payload |= (np.uint64(self.proc_id) & 0xf) << (uci_size - pos)

        # fill the nack             // 1 bit
        pos += 1
        self.payload |= (np.uint64(self.ACK_NACK) & 1) << (uci_size - pos)

        # fill the SR               // 1 bit
        pos += 1
        self.payload |= (np.uint64(self.SR) & 1) << (uci_size - pos)

        # fill the CSI              // 4 bits
        pos += 4
        self.payload |= (np.uint64(self.CSI) & 0xf) << (uci_size - pos)
