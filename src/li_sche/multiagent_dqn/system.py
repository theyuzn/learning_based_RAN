import argparse
from random import randrange
from .envs.env import RAN_system
from .agent import Agent
import li_sche.utils.pysctp.sctp as sctp

class System():
    ### Init
    def __init__(self, args: argparse.Namespace, gNB_sock : sctp.sctpsocket_tcp, conn_sock : sctp.sctpsocket_tcp):
        # Agent
        self.agent = Agent(args = args)
        self.env = RAN_system(args = args)
        self.gNB_sock = gNB_sock
        self.ue_sock = conn_sock

    ### Get initial states
    def get_initial_states(self):
        state = self.env.reset()
        return state

    ############################### Test ############################## 
    def test_system(self):
        state = self.env.reset()
        slot = 0
        done = False
        while not done:
            ul_uelist = state.ul_uelist

            used_rb = 0
            if state.schedule_slot_info == 'U':
                for i in range(len(ul_uelist)):
                    ul_uelist[i].set_Group(randrange(4))
                    ul_uelist[i].set_RB(1)
                    used_rb += 1
                    # print(rf'used_rb ${used_rb}')
                    if used_rb > 246:
                        break

            state, reward, done = self.env.step(action = ul_uelist)
            slot += 1
            print(f'{state}, {reward}, {done}', end = '\n')
    ###################################################################


    ############################# Train ###############################
    def train(self):
        self.agent.train()
    ###################################################################


    ############################# Train ###############################
    def PF(self):
        return
    ###################################################################

    ############################# Train ###############################
    def RR(self):
        return
    ###################################################################

    # def save_checkpoint(self, filename='dqn_checkpoints/checkpoint.pth.tar'):
    #     dirpath = os.path.dirname(filename)

    #     if not os.path.exists(dirpath):
    #         os.mkdir(dirpath)

    #     checkpoint = {
    #         'dqn': self.dqn.state_dict(),
    #         'target': self.target.state_dict(),
    #         'optimizer': self.optimizer.state_dict(),
    #         'step': self.step,
    #         'best': self.best_score,
    #         'best_count': self.best_count
    #     }
    #     torch.save(checkpoint, filename)

    # def load_checkpoint(self, filename='dqn_checkpoints/checkpoint.pth.tar', epsilon=None):
    #     checkpoint = torch.load(filename)
    #     self.dqn.load_state_dict(checkpoint['dqn'])
    #     self.target.load_state_dict(checkpoint['target'])
    #     self.optimizer.load_state_dict(checkpoint['optimizer'])
    #     self.step = checkpoint['step']
    #     self.best_score = self.best_score or checkpoint['best']
    #     self.best_count = checkpoint['best_count']

    # def load_latest_checkpoint(self, epsilon=None):
    #     r = re.compile('chkpoint_(dqn|lstm)_(?P<number>-?\d+)\.pth\.tar$')

    #     files = glob.glob(f'dqn_checkpoints/chkpoint_{self.mode}_*.pth.tar')

    #     if files:
    #         files = list(map(lambda x: [int(r.search(x).group('number')), x], files))
    #         files = sorted(files, key=lambda x: x[0])
    #         latest_file = files[-1][1]
    #         self.load_checkpoint(latest_file, epsilon=epsilon)
    #         print(f'latest checkpoint has been loaded - {latest_file}')
    #     else:
    #         print('no latest checkpoint')


    # @property
    # def play_step(self):
    #     return np.nan_to_num(np.mean(self._play_steps))

    # def _sum_params(self, model):
    #     return np.sum([torch.sum(p).data[0] for p in model.parameters()])

    # def imshow(self, sample_image: np.array, transpose=False):
    #     if transpose:
    #         sample_image = sample_image.transpose((1, 2, 0))
    #     pylab.imshow(sample_image, cmap='gray')
    #     pylab.show()

