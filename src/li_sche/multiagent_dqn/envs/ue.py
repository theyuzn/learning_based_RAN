class UE:
    def __init__(self, id, sizeOfData, delay_bound, type, group = -1, rb_id = -1):
        self.id = id
        self.sizeOfData = sizeOfData
        self.delay_bound = delay_bound      
        self.type=type  
        self.group = group
        self.rb_id = rb_id


    def set_RB_ID(self, rb_id):
        self.rb_id = rb_id

        
        