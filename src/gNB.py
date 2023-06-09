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

from li_sche.multiagent_dqn.algorithms import System

parser = argparse.ArgumentParser(description='Configuration')
########################################## RAN parameter ##########################################
parser.add_argument('--bw', default=400, type=int,help='channel bandwidth in MHz')
parser.add_argument('--mu', default=3,   type=int,help='the numerology')
parser.add_argument('--rb', default=264,type=int,help='number of available RB')

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
SERVER_HOST="172.17.0.2"
MAX_BUFFER_SIZE = 65535


from collections import deque
def main(parser: argparse.Namespace):

    # Test Block # 
    
    ##############
    '''
    To emulate the Rx and Tx, I create two socket
    recv_sock is responsible for the Rx antenna
    send_sock is responsible for the Tx antenna
    The recv_sock will ocuupies one thread to receive the msg.
    '''
    recv_sock : sctp.sctpsocket_tcp = 0
    send_sock : sctp.sctpsocket_tcp = 0

    # Create the server socket, to bind and wait for connections
    # For Rx antenna
    server_recv_sock = sctp.sctpsocket(socket.AF_INET, socket.SOCK_STREAM, None)
    server_recv_sock.initparams.max_instreams = 1
    server_recv_sock.initparams.num_ostreams = 1
    server_recv_sock.bindx([(SERVER_HOST, RECV_PORT)])
    server_recv_sock.listen(5)
    server_recv_sock.events.data_io = 1
    server_recv_sock.events.clear()
    print("Waiting for UE's Tx conntenion")
    recv_sock, _ = server_recv_sock.accept()
    print(f"recv_sock is connected.\nUplink channel is established.")
    print("=========================================")


    # For Tx antenna
    server_send_sock = sctp.sctpsocket(socket.AF_INET, socket.SOCK_STREAM, None)
    server_send_sock.initparams.max_instreams = 1
    server_send_sock.initparams.num_ostreams = 1
    server_send_sock.bindx([(SERVER_HOST, SEND_PORT)])
    server_send_sock.listen(5)
    server_send_sock.events.data_io = 1
    server_send_sock.events.clear()

    print("Waiting for UE's Rx conntenion")
    send_sock, _ = server_send_sock.accept()
    print(f"send_sock is connected.\nDownlink channel is established.")
    print("=========================================")

    
    
    # Initial the system
    system = System(args = parser, send_sock = send_sock, recv_sock = recv_sock)
 
    # Running the system
    scheduler = parser.scheduler
    match scheduler:
        # Perform testing in the system
        case "Test":
            system.FCFS()
            return
        
        case "test":
            system.FCFS()
        
        ## Perform lightweight scheduler using DQN
        case "Li":
            system.train()

        ## Perform Proportional Fairness algorithm
        case "PF":
            system.PF()
            return
        
        ## Perform Round Robin algorithm
        case "RR":
            system.RR()
            return


    time.sleep(1)
    recv_sock.close()
    send_sock.close()
    server_recv_sock.close()
    server_send_sock.close()
    print("Process is done, BYE!!!")
       
if __name__ == '__main__':
    main(parser)