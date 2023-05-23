import socket
import os
import li_sche.utils.pysctp.sctp
import json
import socket
import sctp
from li_sche.multiagent_dqn.envs.ue import UE
import math

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
                sizeOfData      = dct[SIZE],
                delay_bound     = dct[DELAY]*math.pow(2, 1),
                window          = dct[WINDOW],
                service         = dct[SERVICE],
                nr5QI           = dct[NR5QI],
                errorrate       = dct[ER],
                type            = dct[TYPE],
                group           = -1,
                rb_id           = -1,
                nrofRB          = -1,
                is_SR_sent      = False)

# class UE_entity(UE):
    

def main():
    sk = sctp.sctpsocket_tcp(socket.AF_INET)
    sk.connect(("172.17.0.2", 3333))
    sk.sctp_send("test123")

    UE_List = list()
    cur_path = os.path.dirname(__file__)
    new_path = os.path.join(cur_path, '../../../data/uedata.json')
    with open(new_path, 'r') as ue_file:
        Global_UE_list = json.load(ue_file,object_hook=decode_json) 
    for i in range(len(Global_UE_list)):
        Global_UE_list[i].id = i + 1
            






    sk.close()

if __name__ == '__main__':
    main()