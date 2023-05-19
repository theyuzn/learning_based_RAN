import argparse
from li_sche.multiagent_dqn.system import System
import socket
import li_sche.utils.pysctp.sctp as sctp

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


SERVER_PORT=3333
SERVER_HOST="172.17.0.2"

def main(parser: argparse.Namespace):  

    # Initial the system
    system = System(args = parser)

    # Create the sctp socket
    addr = (SERVER_HOST, SERVER_PORT)
    gNB_sock = sctp.sctpsocket(socket.AF_INET, socket.SOCK_STREAM, None)
    gNB_sock.initparams.max_instreams = 3
    gNB_sock.initparams.num_ostreams = 3
    gNB_sock.bindx([addr])
    gNB_sock.listen(5)
    gNB_sock.events.data_io = 1
    gNB_sock.events.clear()

    conn_sock : sctp.sctpsocket_tcp
    while True:
        print("Waiting for user connecting")
        conn_sock, addr = gNB_sock.accept()
        print(f"connecting: {addr}")
        break

    # Running the system
    scheduler = parser.scheduler
    match scheduler:
        # Perform testing in the system
        case "test":
            system.test_system()
            return
        
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

    conn_sock.close()
    gNB_sock.close()

if __name__ == '__main__':
    main(parser)