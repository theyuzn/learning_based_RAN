'''
Creater Chuang, Yu-Hsin
Lab : MWNL -- Mobile & Wireless Networking Labtory
Advisor : S.T. Sheu
Copyright @Brandon, @Yu-Hsin Chuang, @Chuang, Yu-Hsin.
All rights reserved.

Created in 2023/03
'''


import socket
import os
import li_sche.utils.pysctp.sctp as sctp
import json
import socket
import math
import numpy as np
import time
from li_sche.multiagent_dqn.envs.thread import Socket_Thread

from li_sche.multiagent_dqn.envs.ue import UE
from li_sche.multiagent_dqn.envs.msg import *
import  li_sche.multiagent_dqn.envs.msg as MSG_HDR


##################################################################
# UE data constant
ID          = 'id'
SIZE        = 'sizebyte'
DELAY       = 'delayms'
WINDOW      = 'avgwindowms'
SERVICE     = 'service'
NR5QI       = '5QI'
ER          = 'errorrate'
TYPE        = 'type'

# Constant        
BUFFERSIZE = 65535

# Global variables ##
system_frame = 0
system_slot = 0
system_k0 = 0
system_k1 = 0
system_k2 = 0
done = False
UE_List = list()
i = 0
##################################################################
def LOG(log):
    print(f"[UE] --> {log}")

def decode_json(dct):
    global i
    i += 1
    return  UE(
                id              = i, 
                total_data      = dct[SIZE],
                rdb             = dct[DELAY]*math.pow(2, 1),
                service         = dct[WINDOW],
                nr5QI           = dct[NR5QI],
                errorrate       = dct[ER],
                type            = dct[TYPE])


def decode_msg(self, msg : np.uint64):
    pass

def downlink_channel(msg : bytes):
    recv_msg = MSG()
    msg = int.from_bytes(msg, "big")
    recv_msg.payload = msg
    header = recv_msg.decode_header()

    global system_frame, system_slot, system_k0, system_k1, system_k2, UE_List, done
    match header:
        case MSG_HDR.HDR_INIT:
            LOG("Initial is about to processing")
            system_frame = 0
            system_slot = 0
            system_k0 = 0
            system_k2 = 0
            done = False

            # Load the UEs
            UE_List = list()
            path = f"{os.path.dirname(__file__)}/data/uedata.json"
            with open(path, 'r') as ue_file:
                UE_List = json.load(ue_file,object_hook=decode_json) 

            # Unpack the Init msg
            init_msg = INIT()
            init_msg.payload = msg
            init_msg.header = header
            init_msg.decode_msg()
            system_k0 = init_msg.k0
            system_k1 = init_msg.k1
            system_k2 = init_msg.k2
            LOG(f"RRC connection ==> k0 = {system_k0}, k2 = {system_k2}")
            LOG("Initial is done")

        case MSG_HDR.HDR_END:
            done = True

        case MSG_HDR.HDR_SYNC:
            sync_msg = SYNC()
            sync_msg.payload = msg
            sync_msg.header = header
            sync_msg.decode_msg()
            system_frame = sync_msg.frame
            system_slot = sync_msg.slot
            LOG(f"Slot indication ==> Frame : {system_frame}, Slot : {system_slot}")

        case _:
            
            pass

def uplink_channel(sock : sctp.sctpsocket_tcp, msg : np.uint64):
    msg = int(msg).to_bytes()
    sock.sctp_send(msg)

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
    send_sock.connect(("172.17.0.3", 3333))
    time.sleep(1)
    # The recv_sock needs to connect to the send_sock in the gNB
    recv_sock = sctp.sctpsocket_tcp(socket.AF_INET)
    recv_sock.connect(("172.17.0.3", 3334))


    receive_thread = Socket_Thread(name = "DL_Thread", socket = recv_sock, callback = downlink_channel)
    receive_thread.start()
    

    ## Initial ##
    global system_frame, system_slot, system_k0, system_k1, system_k2, UE_List, done
   

    while not done:

        continue
        

   

            
    time.sleep(1)
    send_sock.close()
    recv_sock.close()
    LOG("Process is done, BYE!!!")

if __name__ == '__main__':
    main()