class UE:
    def __init__(self, 
                 id,            
                 total_data,    
                 rdb,   
                 service,       
                 nr5QI,        
                 errorrate,     
                 type):
        # ----[==================================]---- #
        # ----[         Common UE Config         ]---- #
        # ----[   Including the statistic data   ]---- #
        # ----[==================================]---- #
        self.id = id                    # UE ID (can be seemed as RNTI)
        self.total_data = total_data    # The total size of data
        self.bsr = 0                    # Buffer Status Report
        self.rdb = rdb                  # Remaining delay budget
        self.service = service          # String [VoNR, Video, Live Stream]
        self.nr5QI = nr5QI              # 5QI
        self.errorrate = errorrate      # Errorrate correspond to 5QI
        self.type=type                  # mMTC for now
        self.queuing_delay = list()     # To store the queuing delay (For each packet)


        # ----[==================================]---- #
        # ----[      The scheduliong result      ]---- #
        # ----[          Carried in DCI          ]---- #
        # ----[==================================]---- #
        #             [ Original DCI msg ]             #
        self.mcs = mcs                  # MCS
        self.start_symbol = 0           # Start symbol
        self.time_length = 0            # Time duration
        self.start_rb = 0               # Start RB [RB index]
        self.freq_leng = 0              # RB size
        #             [ Proposed DCI msg ]             #
        self.contention = False         # Contention flag [0 : schedule; 1 contention]
        self.contention_size = 0        # The number of RB each contention UE can access
        

        # ----[==================================]---- #
        # ----[   Perform Contention Parameter   ]---- #
        # ----[==================================]---- #
        self.rb_id = rb_id              # In order to perform contention


