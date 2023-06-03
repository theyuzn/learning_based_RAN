import socket
import os
import li_sche.utils.pysctp.sctp
import json
import socket
import sctp
import math
import numpy as np
import time
from li_sche.multiagent_dqn.envs.thread import Socket_Thread

from li_sche.multiagent_dqn.envs.ue import UE
from li_sche.multiagent_dqn.envs.msg import MSG

# UE data constant
ID          = 'id'
SIZE        = 'sizebyte'
DELAY       = 'delayms'
WINDOW      = 'avgwindowms'
SERVICE     = 'service'
NR5QI       = '5QI'
ER          = 'errorrate'
TYPE        = 'type'


def decode_json(dct):
    return  UE(
                id              = 0, 
                total_data      = dct[SIZE],
                rdb             = dct[DELAY]*math.pow(2, 1),
                service         = dct[WINDOW],
                nr5QI           = dct[NR5QI],
                errorrate       = dct[ER],
                type            = dct[TYPE])

# class UE_entity(UE):
BUFFERSIZE = 65535

def callback(msg : bytes = bytes(1)):
    print(msg)

def main():
    '''
    To emulate the Rx and Tx, create two socket
    recv_sock is responsible for the Rx antenna
    send_sock is responsible for the Tx antenna
    The recv_sock will ocuupies one thread to receive the msg.
    '''

    recv_sock : sctp.sctpsocket_tcp = 0
    send_sock : sctp.sctpsocket_tcp = 0

    # The send_sock needs to connect to the recv_sock in the gNB
    send_sock = sctp.sctpsocket_tcp(socket.AF_INET)
    send_sock.connect(("172.17.0.2", 3333))
    time.sleep(1)
    # The recv_sock needs to connect to the send_sock in the gNB
    recv_sock = sctp.sctpsocket_tcp(socket.AF_INET)
    recv_sock.connect(("172.17.0.2", 3334))


    receive_thread = Socket_Thread(name = "UE_thread", socket = recv_sock, callback = callback)
    receive_thread.run()
    

    ## Initial ##
    '''
    The Global_UE_List is to store the UE data.
    The UE_List is the actual list to perform the system running.
    '''

    UE_List = list()
    Global_UE_List = list()
    cur_path = os.path.dirname(__file__)
    new_path = os.path.join(cur_path, 'data/uedata.json')
    with open(new_path, 'r') as ue_file:
        Global_UE_List = json.load(ue_file,object_hook=decode_json) 
    for i in range(len(Global_UE_List)):
        Global_UE_List[i].id = i + 1
     
    UE_List = Global_UE_List.copy()

   

            


    send_sock.close()
    recv_sock.close()

if __name__ == '__main__':
    main()