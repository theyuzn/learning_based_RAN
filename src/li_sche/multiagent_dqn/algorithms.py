'''
Creater Chuang, Yu-Hsin
Lab : MWNL -- Mobile & Wireless Networking Labtory
Advisor : S.T. Sheu
Copyright @Brandon, @Yu-Hsin Chuang, @Chuang, Yu-Hsin.
All rights reserved.

Created in 2023/03
'''

import argparse
import li_sche.utils.pysctp.sctp as sctp
import time
import math

from collections import deque, namedtuple
from random import randrange
from .agent import Agent
from .envs.env import RAN_system, Schedule_Result
from .envs.msg import *
from .envs.ue import UE 

class Algorithms():
    def __init__(self, args: argparse.Namespace):
        # Agent
        self.agent = Agent(args = args)
        self.env = RAN_system(args = args)
        self.pusch_result_transition = namedtuple('PUSCH_RESULT', ('frame', 'slot', 'nrof_UE', 'cumulated_rb'))


    # Schedule the PUSCH and Return the DCI0
    def schedule_pusch(self, frame, slot, ul_queue : deque):
        dci0 = list()
        pusch = self.pusch_result_transition(frame = frame, slot = slot, nrof_UE = 0, cumulated_rb = 0)

        # Only schedule in DL slot due to the DCI0
        current_slot = self.env.get_slot_info(frame = frame, slot = slot)
        if current_slot != 'D':
            return dci0, pusch
        
        contention_size = 0
        schedule_size = 0
        scheduled_UE = deque([], maxlen=65535)
        contention_UE = deque([], maxlen=65535)

        # To cound the size of each scheme
        while len(ul_queue) > 0 :
            ul_ue : UE = ul_queue.popleft()
            ul_ue.freq_len = ul_ue.bsr  
            
            if ul_ue.rdb <= 5 * self.env.pattern_p:
                # Schedule
                ul_ue.contention = False
                schedule_size += ul_ue.freq_len
                scheduled_UE.append(ul_ue)
                
            else:
                ul_ue.contention = True
                contention_size += ul_ue.freq_len
                contention_UE.append(ul_ue)
            
            if schedule_size + contention_size >= self.env.nrofRB:
                if ul_ue.contention:
                    ul_queue.appendleft(contention_UE.pop())
                else:
                    ul_queue.appendleft(scheduled_UE.pop())
                break
        

        # Fill PUSCH result
        cumulated_rb = schedule_size + contention_size
        nrof_UE  = len(scheduled_UE) + len(contention_UE)

        # To fill DCI for scheduled UE
        start_rb = 0
        while len(scheduled_UE) > 0:
            ue : UE = scheduled_UE.pop()
            
            dci = DCI_0_0()
            dci.header = ue.id
            dci.UE_id = ue.id
            dci.start_rb = start_rb
            dci.freq_len = ue.freq_len
            
            start_rb += ue.freq_len
            dci.fill_payload()
            dci0.append(dci.payload)
        
        # To fill DCI for shared UE
        while len(contention_UE) > 0:
            ue : UE = contention_UE.pop()
            
            dci = DCI_0_0()
            dci.header = ue.id
            dci.UE_id = ue.id
            dci.start_rb = start_rb
            dci.freq_len = ue.freq_len
            dci.contention = True
            dci.contention_size = contention_size

            dci.fill_payload()
            dci0.append(dci.payload)

        pusch = self.pusch_result_transition(frame = frame, slot = slot+self.env.k2, nrof_UE = nrof_UE ,cumulated_rb = cumulated_rb)
        return dci0, pusch
    

    def schedule_downlink(self, frame, slot):
        current_slot = self.env.get_slot_info(frame = frame, slot = slot)
        # Only schedule in Special slot
        if current_slot != 'S':
            return None
        
        dci : DCI_1_0 = DCI_1_0()
        dci.UE_id = 16
        dci.fill_payload()
        return [dci.payload]


    ############################### FCFS ############################## 

    def FCFS(self):
        print("\n\n[ This is the First-Come-First-Serve ]\n\n")
        count = 0

        while count < 10000 :
            # To initial the state
            state_tuple, reward, done = self.env.init_RAN_system()
            ul_queue = deque([], maxlen = 65535) # Max number of the UE
            cumulative_reward = reward
            
            while not done:
                # initial
                action = Schedule_Result()  
                ul_req = deque([], maxlen = 65535)

                # Update the system parameters
                frame = state_tuple.frame
                slot = state_tuple.slot
                ul_req = deque(state_tuple.ul_req, maxlen = 65535)

                for ue in ul_queue:
                    ue.rdb -= 1
                
                # Add request into queue
                for ue in ul_req:
                    ul_queue.append(ue)
                ul_req.clear()
            
                # Schedule the DCI0 and UL Data sequentially
                dci0, pusch_result = self.schedule_pusch(frame = frame, slot = slot, ul_queue = ul_queue) 

                # Schedule the DCI1 for informing the UCCH
                dci1 = self.schedule_downlink(frame = frame, slot = slot)
            
                action.DCCH = action.DCI_Transition(dci0 = dci0, dci1 = dci1)
                action.USCH = action.USCH_Transition(frame = pusch_result.frame, slot = pusch_result.slot, nrof_UE = pusch_result.nrof_UE, cumulatied_rb = pusch_result.cumulated_rb)
                
                state_tuple, reward, done = self.env.step(action)
                cumulative_reward += reward
                time.sleep(0.001)
            print(cumulative_reward)
            
    ###################################################################


    ############################# Train ###############################
    def train(self):
        self.agent.train()
    ###################################################################


    def proportional_fair(users, channel_qualities, previous_throughputs, alpha):
        priorities = []
        for i, user in enumerate(users):
            priority = (1 - alpha) * math.log(1 + channel_qualities[i]) + alpha * previous_throughputs[i]
            priorities.append(priority)

        selected_user = max(zip(users, priorities), key=lambda x: x[1])[0]
        return selected_user


    ############################# Train ###############################
    def PF(self):
        
        return
    ###################################################################

    def round_robin(users, current_user_index):
        selected_user = users[current_user_index]
        current_user_index = (current_user_index + 1) % len(users)
        return selected_user, current_user_index


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




# Example usage
# users = [1, 2, 3, 4, 5]
# channel_qualities = [0.8, 0.9, 0.7, 0.6, 0.8]
# previous_throughputs = [10, 15, 12, 9, 11]
# alpha = 0.5

# # Proportional Fair Algorithm
# selected_user = proportional_fair(users, channel_qualities, previous_throughputs, alpha)
# print("Proportional Fair Algorithm - Selected User:", selected_user)

# # Round Robin Algorithm
# current_user_index = 0
# selected_user, current_user_index = round_robin(users, current_user_index)
# print("Round Robin Algorithm - Selected User:", selected_user)

# # First-Come-First-Serve Algorithm
# selected_user = first_come_first_serve(users)
# print("First-Come-First-Serve Algorithm - Selected User:", selected_user)
