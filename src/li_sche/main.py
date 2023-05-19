import argparse

from multiagent_dqn.system import System

parser = argparse.ArgumentParser(description='Configuration')
########################################## RAN parameter ##########################################
parser.add_argument('--bw', default=400,type=int,help='channel bandwidth in MHz')
parser.add_argument('--mu', default=3, type=int,help='the numerology')
parser.add_argument('--rb', default=264,type=int,help='number of available RB')
parser.add_argument('--k0', default=0, type=int, help='k0 parameter')
parser.add_argument('--k1', default=2, type=int, help='k1 parameter')
parser.add_argument('--k2', default=4, type=int, help='k2 parameter')

########################################## DQN parameter ##########################################
parser.add_argument('--model', default='dqn', type=str, help='forcefully set step')
parser.add_argument('--step', default=None, type=int, help='forcefully set step')
parser.add_argument('--best', default=None, type=int, help='forcefully set best')
parser.add_argument('--load_latest', dest='load_latest', action='store_true', help='load latest checkpoint')
parser.add_argument('--no_load_latest', dest='load_latest', action='store_false', help='train from the scrach')
parser.add_argument('--checkpoint', default=None, type=str, help='specify the checkpoint file name')
parser.add_argument('--mode', dest='mode', default='play', type=str, help='[play, train]')
parser.add_argument('--game', default='FlappyBird-v0', type=str, help='only Pygames are supported')
parser.add_argument('--clip', dest='clip', action='store_true', help='clipping the delta between -1 and 1')
parser.add_argument('--noclip', dest='clip', action='store_false', help='not clipping the delta')
parser.add_argument('--skip_action', default=4, type=int, help='Skipping actions')
parser.add_argument('--record', dest='record', action='store_true', help='Record playing a game')
parser.add_argument('--inspect', dest='inspect', action='store_true', help='Inspect CNN')
parser.add_argument('--seed', default=111, type=int, help='random seed')
parser.add_argument('--repeat', default = 4, type = int, help = 'The LSTM model')

########################################## Exp parameter ##########################################
parser.add_argument('--test', default=False, type = bool, help='To test the RAN system')
parser.add_argument('--scheduler', default="Li", help='To indicate the scheduler algorithm')

##########################################  Set default  ##########################################
parser.set_defaults(test = False, scheduler = "Li")
parser: argparse.Namespace = parser.parse_args()


class TT():
    def __init__(self):
        self.test = 0

def main(parser: argparse.Namespace):     
    
    # repeat_action = parser.repeat
    # test = parser.test
    # scheduler = parser.scheduler
    
    # system = System(args = parser, cuda = True, action_repeat = repeat_action)
    # if test:
    #     system.test_system()
    #     return

    # match scheduler:

    #     ## Perform lightweight scheduler using DQN
    #     case "Li":
    #         system.train()

    #     ## Perform Proportional Fairness algorithm
    #     case "PF":
    #         return
        
    #     ## Perform Round Robin algorithm
    #     case "RR":
    #         return

if __name__ == '__main__':
    main(parser)