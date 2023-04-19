class UE:
    def __init__(self, id, sizeOfData, delay_bound, window, service, nr5QI, errorrate, type, group = -1, nrofRB = 0, mcs = 0, rb_id = -1, is_SR_sent = False):
        self.id = id
        self.sizeOfData = sizeOfData
        self.delay_bound = delay_bound   
        self.window = window
        self.service = service
        self.nr5QI = nr5QI
        self.errorrate = errorrate   
        self.type=type  
        self.group = group
        self.nrofRB = nrofRB
        self.mcs = mcs
        self.rb_id = rb_id
        self.is_SR_sent = is_SR_sent


    def set_Group(self, group_id : int):
        self.group = group_id
    
    def set_RB(self, nrofRB : int):
        self.nrofRB = nrofRB

    def set_RB_ID(self, rb_id : int):
        self.rb_id = rb_id

    def set_SR(self, SR : bool):
        self.is_SR_sent = SR

    def decay_delay(self, slot):
        self.delay_bound -= slot

    def decay_size(self, size):
        self.sizeOfData -= size

    def set_mcs(self, mcs):
        self.mcs = mcs

    def get_UE_ID(self):
        return self.id
