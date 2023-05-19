class UE:
    def __init__(self, 
                 id,            
                 sizeOfData,    
                 delay_bound,   
                 service,       
                 nr5QI,        
                 errorrate,     
                 type,         
                 mcs = 0, 
                 rb_id = -1):
        # ----[==================================]---- #
        # ----[         Common UE Config         ]---- #
        # ----[==================================]---- #
        self.id = id                    # UE ID (can be seemed as RNTI)
        self.sizeOfData = sizeOfData    # The total size of UL data
        self.delay_bound = delay_bound  # Remaining delay budget
        self.service = service          # String [VoNR, Video, Live Stream]
        self.nr5QI = nr5QI              # 5QI   
        self.errorrate = errorrate      # Errorrate correspond to 5QI
        self.type=type                  # mMTC for now

        # ----[==================================]---- #
        # ----[      The scheduliong result      ]---- #
        # ----[          Carried in DCI          ]---- #
        # ----[==================================]---- #
        # Dedecated DCI msg
        self.mcs = mcs                  # MCS
        self.start_symbol = 0           # Start symbol
        self.time_length = 0            # Time duration
        self.start_rb = 0               # Start RB [RB index]
        self.freq_leng = 0              # RB size
        # Proposed DCI msg
        self.contention = False         # Contention flag [0 : schedule; 1 contention]
        self.RB_size = 0                # The number of RB each contention UE can access
        
        # ----[==================================]---- #
        # ----[   Perform Contention Parameter   ]---- #
        # ----[==================================]---- #
        self.rb_id = rb_id              # In order to perform contention
