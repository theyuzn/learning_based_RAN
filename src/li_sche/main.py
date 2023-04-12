import argparse

from  multiagent_dqn.envs.env import Env

parser = argparse.ArgumentParser(description='DQN Configuration')
parser.add_argument('--no_load_latest', dest='load_latest', action='store_false', help='train from the scrach')
parser.add_argument('--checkpoint', default=None, type=str, help='specify the checkpoint file name')
parser.add_argument('--mode', dest='mode', default='run', type=str, help='[run, train]')
parser.add_argument('--record', dest='record', action='store_true', help='Record playing a game')
parser.add_argument('--seed', default=111, type=int, help='random seed')
parser.add_argument('--bw', default=40,type=int,help='channel bandwidth in MHz')
parser.add_argument('--mu', default=2, type=int,help='the numerology')
parser.add_argument('--rb', default=106,type=int,help='number of available RB')
parser.add_argument('--k0', default=0, type=int, help='k0 parameter')
parser.add_argument('--k1', default=2, type=int, help='k1 parameter')
parser.add_argument('--k2', default=4, type=int, help='k2 parameter')
parser: argparse.Namespace = parser.parse_args()


def main(parser: argparse.Namespace):
    env = Env(args = parser)

if __name__ == '__main__':
    main(parser)