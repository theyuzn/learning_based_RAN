import socket
import os
import li_sche.utils.pysctp.sctp
import json
import socket
import sctp
import math
import numpy as np

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

def main():
    
    tt : np.uint64 = 0x15600000
    print("0x%x" %tt)

    tt |= 1<<3
    tt = 0b11111111
    test = 0xff000000
    print(tt)
    print(test >> 28)
    print((test>>24) == tt)
    t = test & tt
    print("0x%x  %x" %(t,tt))
    return

    sk = sctp.sctpsocket_tcp(socket.AF_INET)
    sk.connect(("172.17.0.2", 3333))
    sk.sctp_send("test123")

    UE_List = list()
    cur_path = os.path.dirname(__file__)
    new_path = os.path.join(cur_path, 'data/uedata.json')
    with open(new_path, 'r') as ue_file:
        Global_UE_list = json.load(ue_file,object_hook=decode_json) 
    for i in range(len(Global_UE_list)):
        Global_UE_list[i].id = i + 1


    while True:
        fromaddr, flags, msg, notif = sk.sctp_recv(BUFFERSIZE)

        print(type(msg))

        match(msg):
            case "Initial":
                print(msg)
                break
            



    sk.close()

if __name__ == '__main__':
    main()