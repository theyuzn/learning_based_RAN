import argparse

class RAN_config:

    def __init__(self, 
                 BW         = 40, 
                 numerology = 1, 
                 nrofRB     = 106, 
                 k0         = 0,
                 k1         = 2,
                 k2         = 4,
                 uelist     = []):
        self.BW         = BW
        self.numerology = numerology
        self.nrofRB     = nrofRB
        self.k0         = k0
        self.k1         = k1
        self.k2         = k2
        self.uelist     = uelist



class Env:

    def __init__(self, args : argparse.Namespace, uelist):
        self.args = args
        self.uelist = uelist

        self.ran = RAN_config(BW = self.args.bw,
                              numerology = self.args.mu,
                              nrofRB = self.args.rb,
                              k0 = self.args.k0,
                              k1 = self.args.k1,
                              k2 = self.args.k2,
                              uelist = self.uelist
                              )
    
    def step(self, action):
        

