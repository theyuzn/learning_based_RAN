'''
Creater Chuang, Yu-Hsin
Lab : MWNL -- Mobile & Wireless Networking Labtory
Advisor : S.T. Sheu
Copyright @Brandon, @Yu-Hsin Chuang, @Chuang, Yu-Hsin.
All rights reserved.

Created in 2023/03
'''


import argparse
import time
import socket
import numpy as np
import li_sche.utils.pysctp.sctp as sctp

from li_sche.multiagent_dqn.algorithms import Algorithms

parser = argparse.ArgumentParser(description='Configuration')
########################################## RAN parameter ##########################################
parser.add_argument('--bw', default=400, type=int,help='channel bandwidth in MHz')
parser.add_argument('--mu', default=3,   type=int,help='the numerology')
parser.add_argument('--rb', default=248,type=int,help='number of available RB')

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
parser.add_argument('--scheduler', default="Li", help='To indicate the scheduler algorithm')

##########################################  Set default  ##########################################
parser.set_defaults(test = False, scheduler = "Li")
parser: argparse.Namespace = parser.parse_args()


RECV_PORT = 3333
SEND_PORT = 3334
SERVER_HOST="172.17.0.3"
MAX_BUFFER_SIZE = 65535



from collections import deque

def test(q = deque([], maxlen=65535)):
    for i in range(len(q)):
        q[i] = 100 + i*2
def main(parser: argparse.Namespace):

    # Test Block # 

    ##############
    '''
    To emulate the Rx and Tx, I create two socket
    recv_sock is responsible for the Rx antenna
    send_sock is responsible for the Tx antenna
    The recv_sock will ocuupies one thread to receive the msg.
    '''
   
    # Initial the system
    alg = Algorithms(args = parser)
 
    # Running the system
    scheduler = parser.scheduler
    scheduler = scheduler.lower()
    match scheduler:
        # Perform testing in the system
        case "fcfs":
            alg.FCFS()
            return
        
        ## Perform lightweight scheduler using DQN
        case "li":
            alg.train()

        ## Perform Proportional Fairness algorithm
        case "pf":
            alg.PF()
            return
        
        ## Perform Round Robin algorithm
        case "RR":
            alg.RR()
            return

    time.sleep(1)
    print("Process is done, BYE!!!")
       
if __name__ == '__main__':
    main(parser)