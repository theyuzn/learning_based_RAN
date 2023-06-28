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
UE_List = list()
UL_UE_List = list()
Allocated_UE = list()
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
    global UE_List, UL_UE_List, Allocated_UE, send_UCI, send_Data, done

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
            LOG(f"Slot indication ==> Frame : {system_frame}, Slot : {system_slot}")

            # UL slot
            cumulated_slot = system_frame * system_spf + system_slot
            if SLOT_PATTERN1[cumulated_slot % PATTERN_P1] == 'U':
                send_Data = True
            

        # For each DCI0 or DCI1
        case _:
            dci : DCI = DCI()
            dci.payload = msg
            dci.header = header
            dci.DCI_format = dci.decode_DCI_foramt()

            match dci.DCI_format:
                case 0:
                    LOG(f"Receive DCI format : {dci.DCI_format} for UE : {dci.header}")
                    dci0_0 : DCI_0_0 = DCI_0_0()
                    dci0_0.header = dci.header
                    dci0_0.payload = dci.payload
                    dci0_0.DCI_format = dci.DCI_format
                    dci0_0.decode_msg()

                    for i in range(len(UL_UE_List)):
                        if dci0_0.UE_id == UL_UE_List[i].id:
                            UL_UE_List[i].freq_leng = dci0_0.frequency_domain_assignment
                            UL_UE_List[i].time_length = dci0_0.time_domain_assignment
                            UL_UE_List[i].transmission_time = (system_frame * system_spf + system_slot) + system_k2

                            if dci0_0.contention == 1:
                                UL_UE_List[i].contention = True
                                UL_UE_List[i].contention_size = dci0_0.contention_size
                    

                case 1:
                    LOG(f"Receive DCI format : {dci.DCI_format} for UE : {dci.header}")
                    dci1_0 : DCI_1_0 = DCI_1_0()
                    dci1_0.header = dci.header
                    dci1_0.payload = dci.payload
                    send_UCI = True
                    while send_UCI:
                        continue
                    
            

def uplink_channel(sock : sctp.sctpsocket_tcp, msg : np.uint64):
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
    global UE_List, UL_UE_List, Allocated_UE, send_UCI, send_Data, done

    while not done:
        # send UCI
        if send_UCI:
            init_msg = INIT()
            init_msg.header = HDR_INIT
            init_msg.fill_payload()
            uplink_channel(sock = send_sock, msg = init_msg.payload)

            uci : UCI = UCI()
            for i in range(16):
                i += 1
                if len(UE_List) > 0:
                    ul_ue : UE
                    ul_ue = UE_List.pop()
                    UL_UE_List.append(ul_ue)
                    uci.UE_id = ul_ue.id
                    uci.SR = 1
                    uci.fill_payload()
                    uplink_channel(sock = send_sock, msg = uci.payload)
                else:
                    break

            end_msg = MSG()
            end_msg.header = HDR_END
            end_msg.fill_payload()
            uplink_channel(sock = send_sock, msg = end_msg.payload)
            send_UCI = False    

        # send UL data
        if send_Data:
            print(len(UL_UE_List))
            init_msg = INIT()
            init_msg.header = HDR_INIT
            init_msg.fill_payload()
            uplink_channel(sock = send_sock, msg = init_msg.payload)

            if len(UL_UE_List) > 0:
                for ul_ue in UL_UE_List:
                    ul_msg : UL_Data = UL_Data()
                    ul_msg.UE_id = ul_ue.id
                    ul_msg.payload_size = ul_ue.bsr
                    ul_msg.bsr = ul_ue.bsr
                    ul_msg.fill_payload()
                    uplink_channel(sock = send_sock, msg = ul_msg.payload)

            end_msg = MSG()
            end_msg.header = HDR_END
            end_msg.fill_payload()
            uplink_channel(sock = send_sock, msg = end_msg.payload)

            send_Data = False 

        continue
        

            
    time.sleep(1)
    send_sock.close()
    recv_sock.close()
    LOG("Process is done, BYE!!!")

if __name__ == '__main__':
    main()