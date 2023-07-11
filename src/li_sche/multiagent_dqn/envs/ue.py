class UE:
    def __init__(self, 
                 id = 0,            
                #  total_data = 0, 
                bsr = 0,   
                rdb = 0,   
                #  service = "VoNR",       
                #  nr5QI = "0",        
                #  errorrate = "0",     
                #  type = "eMBB"
                 ):
        # ----[==================================]---- #
        # ----[         Common UE Config         ]---- #
        # ----[   Including the statistic data   ]---- #
        # ----[==================================]---- #
        self.id = id                      # UE ID (can be seemed as RNTI)
        # self.total_data = total_data    # The total size of data
        self.bsr = bsr                    # Buffer Status Report
        self.init_rdb = rdb               # The re-initial R-PDB
        self.rdb = rdb                    # Remaining delay budget
        # self.service = service          # String [VoNR, Video, Live Stream]
        # self.nr5QI = nr5QI              # 5QI
        # self.errorrate = errorrate      # Errorrate correspond to 5QI
        # self.type=type                  # eMBB for now
        self.queuing_delay = 0            # To store the queuing delay (For each packet)


        # ----[==================================]---- #
        # ----[      The scheduliong result      ]---- #
        # ----[          Carried in DCI          ]---- #
        # ----[==================================]---- #
        #             [ Original DCI msg ]             #
        # self.mcs = 0                    # MCS
        # self.start_symbol = 0           # Start symbol
        # self.time_length = 0            # Time duration
        self.start_rb = 0               # Start RB [RB index]
        self.freq_len = 0               # RB size
        self.transmission_time = 0      # The slot
        #             [ Proposed DCI msg ]             #
        self.contention = False         # Contention flag [0 : schedule; 1 contention]
        self.contention_size = 0        # The number of RB each contention UE can access
        

        # ----[==================================]---- #
        # ----[   Perform Contention Parameter   ]---- #
        # ----[==================================]---- #
        self.rb_id = 0                  # In order to perform contention
        self.send_cnt = 1               # To calculate the average queuing delay
        self.fail_cnt = 0               # To calculate the fail count
        self.suc_cnt = 0                # To calculate the successful cound


