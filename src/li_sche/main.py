import argparse
import json
import os

from ue.ue import UE
# from gNB.scheduler.lightweight_scheduler.dqn import DQNAgent
import utils.constant as CONST
# from src.gNB.gNB import gNB
from random import random, sample


parser = argparse.ArgumentParser(description='DQN Configuration')
parser.add_argument('--model', default='dqn', type=str, help='forcefully set step')
parser.add_argument('--step', default=None, type=int, help='forcefully set step')
parser.add_argument('--best', default=None, type=int, help='forcefully set best')
parser.add_argument('--load_latest', dest='load_latest', action='store_true', help='load latest checkpoint')
parser.add_argument('--no_load_latest', dest='load_latest', action='store_false', help='train from the scrach')
parser.add_argument('--checkpoint', default=None, type=str, help='specify the checkpoint file name')
parser.add_argument('--mode', dest='mode', default='run', type=str, help='[run, train]')
parser.add_argument('--game', default='FlappyBird-v0', type=str, help='only Pygames are supported')
parser.add_argument('--clip', dest='clip', action='store_true', help='clipping the delta between -1 and 1')
parser.add_argument('--noclip', dest='clip', action='store_false', help='not clipping the delta')
parser.add_argument('--skip_action', default=4, type=int, help='Skipping actions')
parser.add_argument('--record', dest='record', action='store_true', help='Record playing a game')
parser.add_argument('--inspect', dest='inspect', action='store_true', help='Inspect CNN')
parser.add_argument('--seed', default=111, type=int, help='random seed')
parser.set_defaults(clip=True, load_latest=True, record=False, inspect=False)
parser: argparse.Namespace = parser.parse_args()


def decode_json(dct):
    return  UE(id = dct[CONST.ID],
                sizeOfData = dct[CONST.SIZE],
                delay_bound = dct[CONST.DELAY],
                type = dct[CONST.TYPE])
    

def main(parser):
    uelist = []
    cur_path = os.path.dirname(__file__)
    new_path = os.path.join(cur_path, 'ue/uedata.json')

    with open(new_path, 'r') as ue_file:
        ue = json.load(ue_file,object_hook=decode_json) 
        uelist.append(ue)


    print(random())

    # gnb = gNB(parser = parser, uelist = uelist)
    # gnb.start()

  

if __name__ == '__main__':
    main(parser)