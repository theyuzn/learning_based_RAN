'''
Creater Chuang, Yu-Hsin
Lab : MWNL -- Mobile & Wireless Networking Labtory
Advisor : S.T. Sheu
Copyright @Brandon, @Yu-Hsin Chuang, @Chuang, Yu-Hsin.
All rights reserved.

Created in 2023/03
'''

## Constant Header
HDR_INIT    = 0b0000000000000000
HDR_CM_DCI  = 0b1111111111111101
HDR_SYNC    = 0b1111111111111110
HDR_END     = 0b1111111111111111

class MSG:
    def __init__(self):
        '''
        1.) DownLink Message
        The payload begins with the header, the msg is identified with this.
        // Common message
        * If the header = 0b0000000000000000 (0), it is the initial message.
        * If the header = 0b1111111111111111 (65535), it is the ending message.
        * If the header = 0b1111111111111110 (65534), it is the sync message.
        * If the header = 0b1111111111111101 (65533), it is the common control message.

        // Dedicated message
        The header = UE_id, no matter what is the DCI0 or DCI1.

        2.) Uplink Message
        * The UCI is focusing on the Scheduling Request (SR) after the header in Special Slot
        * The UL Data is on the UL Slot
        '''        

        self.header : int   = 0b0000000000000000    # 16 bits
        self.payload : int  = 0x0000000000000000    # 64 bits

    def fill_payload(self):
        size = 64
        pos = 0

        pos += 16   # Fill Header
        self.payload |= (self.header & ((1 << 16) - 1)) << (size - pos)



    def decode_header(self):
        size = 64
        pos = 0

        # unpack the header          // 16 bits
        pos += 16
        self.header = (self.payload >> (size - pos)) & ((1 << 16) - 1)
        return self.header
    
# RRC setup msg, only get the needed msg
class INIT(MSG):
    '''
            16         4     4    4      10                 40
    |----------------|----|----|----|----------|---------------------------| 
          header       k0   k1   k2      spf             reserved
    '''
    def __init__(self):
        super(INIT, self).__init__()
        self.k0 : int = 0
        self.k1 : int = 0
        self.k2 : int = 0
        self.spf : int = 0
    
    def fill_payload(self):
        self.header = 0b0000000000000000
        size = 64
        pos = 0

        pos += 16   # Fill Header
        self.payload |= (self.header & ((1 << 16) - 1)) << (size - pos)

        pos += 4    # Fill k0
        self.payload |= (int(self.k0) & 0xf) << (size - pos)

        pos += 4    # Fill k1
        self.payload |= (int(self.k1) & 0xf) << (size - pos)

        pos += 4    # Fill k2
        self.payload |= (int(self.k2) & 0xf) << (size - pos)

        pos += 10    # Fill k2
        self.payload |= (int(self.spf) & 0xf) << (size - pos)

    def decode_msg(self):
        size = 64
        pos = 16    # skip the header

        pos += 4
        self.k0 = (int(self.payload) >> (size - pos)) & 0xf

        pos += 4
        self.k1 = (int(self.payload) >> (size - pos)) & 0xf

        pos += 4
        self.k2 = (int(self.payload) >> (size - pos)) & 0xf

        pos += 10
        self.spf = (int(self.payload) >> (size - pos)) & ((1 << 10) - 1)

    
            
class SYNC(MSG):
    '''
            16            10        8                   30
    |----------------|----------|--------|------------------------------| 
          header        frame      slot               reserved
    '''
    def __init__(self):
        super(SYNC, self).__init__()
        self.frame : int = 0
        self.slot : int = 0

    def fill_payload(self):
      
        self.header = 0b1111111111111110
        size = 64
        pos = 0
        
        pos += 16   # Fill Header
        self.payload |= (self.header & ((1 << 16) - 1)) << (size - pos)

        pos += 10   # Fill frame
        self.payload |= (int(self.frame) & ((1 << 10) - 1)) << (size - pos)

        pos += 8    # Fill slot
        self.payload |= (int(self.slot) & 0xff) << (size - pos)

    def decode_msg(self):
        size = 64
        pos = 16    # skip the header

        pos += 10 
        self.frame = (int(self.payload) >> (size - pos)) & ((1 << 10) - 1)

        pos += 8
        self.slot = (int(self.payload) >> (size - pos)) & 0xff

class DCI(MSG):
    '''
            16                               48
    |----------------|------------------------------------------------| 
          header                             DCI
    '''
    def __init__(self):
        super(DCI, self).__init__()
        self.DCI_format : int = 0   # 0 : UL DCI, 1 : DL DCI

    def decode_DCI_foramt(self):
        dci_size = 64
        pos = 16 # skip the header
        
        pos += 1    
        self.DCI_format = (self.payload >> (dci_size - pos)) & 1

        return self.DCI_format

class DCI_0_0(DCI):
    '''
            16        1   4    4  1   5   1  2  4   2  1 1        16          6
    |----------------|-|----|----|-|-----|-|--|----|--|-|-|----------------|------| 
          header      ^ freq time ^  MCS  ^ rv HARQ ^ SUL^    contention   reserved
                     format       F      NDI       TPC  cont. flag
    '''
    def __init__(self):
        super(DCI_0_0,self).__init__()
        self.UE_id = 0                                  # The 16 bits UE id
        self.format_indicator = 0                       # always 0 for UL
        self.frequency_domain_resource_assignment = 0   # number of RB
        self.time_domain_assignment = 0                 # 
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

        pos += 16
        self.header = self.UE_id
        self.payload |= (int(self.header) & ((1 << 16) - 1)) << (dci_size - pos)

        pos += 1
        self.payload |= (int(0)) << (dci_size - pos)

        pos += 4
        self.payload |= (int(self.frequency_domain_assignment) & 0xf) << (dci_size - pos)

        pos += 4
        self.payload |= (int(self.time_domain_assignment) & 0xf) << (dci_size - pos)

        pos += 1
        self.payload |= (int(self.frequceny_hopping_flag) & 1) << (dci_size - pos)

        pos += 5
        self.payload |= (int(self.mcs) & 0x1f) << (dci_size - pos)

        pos += 1
        self.payload |= (int(self.ndi) & 1) << (dci_size - pos)

        pos +=2
        self.payload |= (int(self.rv) & 0x3) << (dci_size - pos)

        pos += 4
        self.payload |= (int(self.harq_process_number) & 0xf) << (dci_size - pos)

        pos += 2
        self.payload |= (int(self.TPC_command_for_schedule_pusch) & 0x3) << (dci_size - pos)

        pos += 1
        self.payload |= (int(self.SUL_indicator) & 1) << (dci_size - pos)

        pos += 1
        self.payload |= (int(self.contention) & 1) << (dci_size - pos)

        pos += 16
        self.payload |= (int(self.contention_size) & ((1 << 16) - 1)) << (dci_size - pos)

    def decode_msg(self):
        dci_size = 64
        pos = 16        # skip the header

        self.UE_id                          = self.header

        pos += 1
        self.format_indicator               = (self.payload >> (dci_size - pos)) & 1

        pos += 4
        self.frequency_domain_assignment    = (self.payload >> (dci_size - pos)) & 0xf

        pos += 4
        self.time_domain_assignment         = (self.payload >> (dci_size - pos)) & 0xf

        pos += 1
        self.frequceny_hopping_flag         = (self.payload >> (dci_size - pos)) & 1

        pos += 5
        self.mcs                            = (self.payload >> (dci_size - pos)) & 0x1f

        pos += 1
        self.ndi                            = (self.payload >> (dci_size - pos)) & 1

        pos +=2
        self.rv                             = (self.payload >> (dci_size - pos)) & 0x3

        pos += 4
        self.harq_process_number            = (self.payload >> (dci_size - pos)) & 0xf

        pos += 2
        self.TPC_command_for_schedule_pusch = (self.payload >> (dci_size - pos)) & 0x3

        pos += 1
        self.SUL_indicator                  = (self.payload >> (dci_size - pos)) & 1

        pos += 1
        self.contention                     = (self.payload >> (dci_size - pos)) & 1

        pos += 16
        self.contention_size                = (self.payload >> (dci_size - pos)) & ((1 << 16) - 1)
  

class DCI_1_0(DCI):
    '''
            16        1         16         4  1   5   1  2  4   2  2   3   3    4
    |----------------|-|----------------|----|-|-----|-|--|----|--|--|---|---|----| 
          header      ^       freq       time ^  MCS  ^ rv HARQ ^ TPC  ^   k1   ^
                    format                   V2P     NDI        DL   PUCCH  reserved
    '''
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

        pos += 16
        self.header = self.UE_id
        self.payload |= (int(self.header) & ((1 << 16) - 1)) << (dci_size - pos)

        pos += 1
        self.payload |= (int(1)) << (dci_size - pos)

        pos += 16
        self.payload |= (int(self.frequency_domain_assignment) & ((1 << 16) - 1)) << (dci_size - pos)

        pos += 4
        self.payload |= (int(self.time_domain_assignment) & 0xf) << (dci_size - pos)

        pos += 1
        self.payload |= (int(self.vrb_to_prb_mapping) & 1) << (dci_size - pos)

        pos += 5
        self.payload |= (int(self.mcs) & 0x1f) << (dci_size - pos)

        pos += 1
        self.payload |= (int(self.ndi) & 1) << (dci_size - pos)

        pos += 2
        self.payload |= (int(self.rv) & 0x3) << (dci_size - pos)

        pos += 4
        self.payload |= (int(self.harq_process_number) & 0xf) << (dci_size - pos)

        pos += 2
        self.payload |= (int(self.downlink_assignment_index) & 0x3) << (dci_size - pos)

        pos += 2
        self.payload |= (int(self.tpc) & 0x3) << (dci_size - pos)

        pos += 3
        self.payload |= (int(self.pucch_resource) & 0x7) << (dci_size - pos)

        pos += 3
        self.payload |= (int(self.pdsch_to_harq_feedback_timing_indicator) & 0x7) << (dci_size - pos)

    def decode_msg(self):
        dci_size = 64
        pos = 16        # skip the header

        self.UE_id                                      = self.header

        pos += 1
        self.format_indicator                           = (self.payload >> (dci_size - pos)) & 1

        pos += 16
        self.frequency_domain_assignment                = (self.payload >> (dci_size - pos)) & ((1 << 16) - 1)

        pos += 4
        self.time_domain_assignment                     = (self.payload >> (dci_size - pos)) & 0xf

        pos += 1
        self.vrb_to_prb_mapping                         = (self.payload >> (dci_size - pos)) & 1

        pos += 5
        self.mcs                                        = (self.payload >> (dci_size - pos)) & 0x1f

        pos += 1
        self.ndi                                        = (self.payload >> (dci_size - pos)) & 1

        pos += 2
        self.rv                                         = (self.payload >> (dci_size - pos)) & 0x3

        pos += 4
        self.harq_process_number                        = (self.payload >> (dci_size - pos)) & 0xf

        pos += 2
        self.downlink_assignment_index                  = (self.payload >> (dci_size - pos)) & 0x3

        pos += 2
        self.tpc                                        = (self.payload >> (dci_size - pos)) & 0x3

        pos += 3
        self.pucch_resource                             = (self.payload >> (dci_size - pos)) & 0x7

        pos += 3
        self.pdsch_to_harq_feedback_timing_indicator    = (self.payload >> (dci_size - pos)) & 0x7      

# This msg is carried on PUCCH Physical Uplink Control Channel
# [ 3GPP TS 38.212 - 6.3.1.1 / 6.3.2.1]
class UCI(MSG):
    '''
            16          4  1 1   4                    38
    |----------------|----|-|-|----|--------------------------------------| 
          header      HARQ ^ SR CSI                 reserved
                         proc_id
    '''
    def __init__(self):
        super(UCI,self).__init__()
        self.ACK_NACK = 0
        self.proc_id = 0
        self.SR = 0
        self.UE_id = 0
        self.CSI = 0

    def fill_payload(self):
        uci_size = 64
        pos = 0

        pos += 16
        self.header = self.UE_id
        self.payload |= (int(self.header) & ((1 << 16) - 1)) << (uci_size - pos)

        pos += 4
        self.payload |= (int(self.proc_id) & 0xf) << (uci_size - pos)

        pos += 1
        self.payload |= (int(self.ACK_NACK) & 1) << (uci_size - pos)

        pos += 1
        self.payload |= (int(self.SR) & 1) << (uci_size - pos)

        pos += 4
        self.payload |= (int(self.CSI) & 0xf) << (uci_size - pos)

    def decode_msg(self):
        uci_size = 64
        pos = 16        # skip the header

        self.UE_id      = self.header

        pos += 4
        self.proc_id    = (self.payload >> (uci_size - pos)) & 0xf

        pos += 1
        self.ACK_NACK   = (self.payload >> (uci_size - pos)) & 1

        pos += 1
        self.SR         = (self.payload >> (uci_size - pos)) & 1

        pos += 4
        self.SR         = (self.payload >> (uci_size - pos)) & 0xf


class UL_Data(MSG):
    '''
            16            8        8        10               22
    |----------------|--------|--------|----------|----------------------| 
          header      Data Size   BSR       RDB            reserved
    '''
    def __init__(self):
        super(UL_Data, self).__init__()
        self.UE_id : int = 0
        self.payload_size : int = 0 # Along with the BSR index
        self.bsr : int = 0 # short BSR : 5 bits, long BSR : 8 bits
        self.rdb : int = 0 # Remaining Delay Budget

    def fill_payload(self):
        ul_data_size = 64
        pos = 0

        pos += 16
        self.header = self.UE_id
        self.payload |= (int(self.header) & ((1 << 16) - 1)) << (ul_data_size - pos)

        pos += 8
        self.payload |= (int(self.payload_size) & 0xff) << (ul_data_size - pos)

        pos += 8
        self.payload |= (int(self.bsr) & 0xff) << (ul_data_size - pos)

        pos += 10
        self.payload |= (int(self.bsr) & ((1 << 10) - 1)) << (ul_data_size - pos)

    def decode_payload(self):
        ul_data_size = 64
        pos = 16            # skip header

        self.UE_id          = self.header

        pos += 8
        self.payload_size   = (self.payload >> (ul_data_size - pos)) & 0xff

        pos += 8
        self.bsr            = (self.payload >> (ul_data_size - pos)) & 0xff

        pos += 10
        self.rdb            = (self.payload >> (ul_data_size - pos)) & ((1 << 10) - 1)
