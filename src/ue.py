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
import random
from li_sche.multiagent_dqn.envs.thread import Socket_Thread

from li_sche.multiagent_dqn.envs.ue import UE
from li_sche.multiagent_dqn.envs.msg import *
from collections import deque
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
SLOT_PATTERN1       = ['D','D','S','U','U']
PATTERN_P1          = 5

# Global variables ##
system_frame = 0
system_slot = 0
system_k0 = 0
system_k1 = 0
system_k2 = 0
system_spf = 0
done = False
send_UCI = False
send_Data = False
UE_List = deque([], maxlen = 65535)
UL_UE_List = deque([], maxlen = 65535)
DCI0_List = deque([], maxlen = 65535)
i = 0
##################################################################
def LOG(log):
    print(f"[UE] --> {log}")

def decode_json(dct):
    global i
    i += 1
    return  UE(
                id              = i, 
                bsr             = math.ceil(dct[SIZE]/848),
                rdb             = dct[DELAY]*math.pow(2, 1),
                # service         = dct[WINDOW],
                # nr5QI           = dct[NR5QI],
                # errorrate       = dct[ER],
                # type            = dct[TYPE]
                )


# THe DL channel is a non-stoping thread to receive the data from gNB
def downlink_channel(msg : bytes):
    recv_msg = MSG()
    msg = int.from_bytes(msg, "big")
    recv_msg.payload = msg
    header = recv_msg.decode_header()

    global system_frame, system_slot, system_k0, system_k1, system_k2, system_spf
    global UE_List, UL_UE_List, DCI0_List, send_UCI, send_Data, done

    match header:
        case MSG_HDR.HDR_INIT:
            system_frame = 0
            system_slot = 0
            system_k0 = 0
            system_k1 = 0
            system_k2 = 0
            system_spf = 0
            done = False

            # Load the UEs
            path = f"{os.path.dirname(__file__)}/data/uedata.json"
            with open(path, 'r') as ue_file:
                UE_List = deque(json.load(ue_file,object_hook=decode_json), maxlen = 65535)

            # Unpack the Init msg
            init_msg = INIT()
            init_msg.payload = msg
            init_msg.header = header
            init_msg.decode_msg()
            system_k0 = init_msg.k0
            system_k1 = init_msg.k1
            system_k2 = init_msg.k2
            system_spf = init_msg.spf
            LOG(f"RRC connection ==> k0 = {system_k0}, k1 = {system_k1}, k2 = {system_k2} with spf = {system_spf}")
            LOG("Initial is done")

        case MSG_HDR.HDR_END:
            done = True

        case MSG_HDR.HDR_SYNC:
            send_UCI = False
            send_Data = False

            sync_msg = SYNC()
            sync_msg.payload = msg
            sync_msg.header = header
            sync_msg.decode_msg()
            system_frame = sync_msg.frame
            system_slot = sync_msg.slot


            # UL slot send UL data blocking
            cumulated_slot = system_frame * system_spf + system_slot
            LOG(f"{SLOT_PATTERN1[cumulated_slot % PATTERN_P1]} Frame : {system_frame}, Slot : {system_slot}")

            if SLOT_PATTERN1[cumulated_slot % PATTERN_P1] == 'U':
                send_Data = True

            while send_Data:
                continue
            

        # For each DCI0 or DCI1
        case _:
            dci : DCI = DCI()
            dci.payload = msg
            dci.header = header
            dci.DCI_format = dci.decode_DCI_foramt()

            match dci.DCI_format:
                case 0:
                    dci0_0 : DCI_0_0 = DCI_0_0()
                    dci0_0.header = dci.header
                    dci0_0.payload = dci.payload
                    dci0_0.DCI_format = dci.DCI_format
                    dci0_0.decode_msg()
                    DCI0_List.append(dci0_0)
                    LOG(f"Receive DCI format : {dci.DCI_format} for UE : {dci.header}")
                    
                case 1:
                    # LOG(f"Receive DCI format : {dci.DCI_format} for UE : {dci.header}")
                    dci1_0 : DCI_1_0 = DCI_1_0()
                    dci1_0.header = dci.header
                    dci1_0.payload = dci.payload
                    send_UCI = True
                    # Send UCI blocking
                    while send_UCI:
                        continue
                    
            

def uplink_channel(sock : sctp.sctpsocket_tcp, msg : int):
    msg = int(msg).to_bytes(16, "big")
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
    global system_frame, system_slot, system_k0, system_k1, system_k2, system_spf
    global UE_List, UL_UE_List, DCI0_List, send_UCI, send_Data, done

    while not done:
        # send UCI
        if send_UCI:
            init_msg = INIT()
            init_msg.header = HDR_INIT
            init_msg.fill_payload()
            # uplink_channel(sock = send_sock, msg = init_msg.payload)
            
            for i in range(16):
                i += 1
                if len(UE_List) > 0:
                    uci : UCI = UCI()
                    ul_ue : UE
                    ul_ue = UE_List.popleft()
                    UL_UE_List.append(ul_ue)
                    uci.UE_id = ul_ue.id
                    uci.header = ul_ue.id
                    uci.SR = 1
                    uci.fill_payload()
                    print(uci.UE_id)
                    # uplink_channel(sock = send_sock, msg = uci.payload)
                else:
                    break

            end_msg = MSG()
            end_msg.header = HDR_END
            end_msg.fill_payload()
            # uplink_channel(sock = send_sock, msg = end_msg.payload)
            send_UCI = False    

        # send UL data
        if send_Data:
            init_msg = INIT()
            init_msg.header = HDR_INIT
            init_msg.fill_payload()
            # uplink_channel(sock = send_sock, msg = init_msg.payload)

            if len(UL_UE_List) > 0:
                dci0 : DCI_0_0
                for dci0 in DCI0_List:
                    for ul_ue in UL_UE_List:
                        if dci0.UE_id == ul_ue.id:
                            
                            ul_ue.start_rb = dci0.start_rb
                            ul_ue.freq_len = dci0.freq_len
                            ul_ue.contention = dci0.contention
                            if ul_ue.contention:
                                ul_ue.start_rb = dci0.start_rb + random.randrange(dci0.contention_size - 1)

                            ul_msg : UL_Data = UL_Data()
                            ul_msg.UE_id = ul_ue.id
                            ul_msg.payload_size = ul_ue.bsr
                            ul_msg.bsr = ul_ue.bsr
                            ul_msg.start_rb = ul_ue.start_rb
                            ul_msg.freq_len = ul_ue.freq_len
                            ul_msg.fill_payload()
                            # uplink_channel(sock = send_sock, msg = ul_msg.payload)


                for ul_ue in UL_UE_List:
                    dci : DCI_0_0
                    for dci in DCI0_List:
                        if ul_ue.id == dci.UE_id:
                            pass

                    if ul_ue.transmission_time == system_frame * system_spf + system_slot:
                        ul_msg : UL_Data = UL_Data()
                        ul_msg.UE_id = ul_ue.id
                        ul_msg.payload_size = ul_ue.bsr
                        ul_msg.bsr = ul_ue.bsr
                        ul_msg.start_rb = ul_ue.start_rb
                        ul_msg.freq_len = ul_ue.freq_len
                        ul_msg.fill_payload()
                        # uplink_channel(sock = send_sock, msg = ul_msg.payload)

            end_msg = MSG()
            end_msg.header = HDR_END
            end_msg.fill_payload()
            # uplink_channel(sock = send_sock, msg = end_msg.payload)

            send_Data = False 

        continue
        

            
    time.sleep(1)
    send_sock.close()
    recv_sock.close()
    LOG("Process is done, BYE!!!")

if __name__ == '__main__':
    main()