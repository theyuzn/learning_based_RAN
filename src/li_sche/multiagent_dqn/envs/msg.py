# This msg is carried on PDCCH Physical Downlink Control Channel
def DCI():
    def __init__(self):
        self.format = 0
        self.freq = 0
        self.time = 0
        self.freq_hopping = 0
        self.MCS = 0
        self.NDI = 0
        self.rv = 0
        self.harq_proc = 0
        self.TCP = 0
        self.SUL_ind = 0



# This msg is carried on PUCCH Physical Uplink Control Channel
# [ 3GPP TS 38.212 - 6.3.1.1 / 6.3.2.1]
def UCI():
    def __init__(self):
        self.ANK_NACK = 0
        self.SR = 0
        self.CSI = 0
