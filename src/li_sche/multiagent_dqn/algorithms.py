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

from collections import deque
from random import randrange
from .agent import Agent
from .envs.env import RAN_system, Schedule_Result
from .envs.msg import *
from .envs.ue import UE 

class System():
    def __init__(self, args: argparse.Namespace, send_sock : sctp.sctpsocket_tcp, recv_sock : sctp.sctpsocket_tcp):
        # Agent
        self.agent = Agent(args = args, send_sock = send_sock, recv_sock = recv_sock)
        self.env = RAN_system(args = args, send_sock = send_sock, recv_sock = recv_sock)



    # Schedule the PUSCH and Return the DCI0
    def schedule_pusch(self, frame, slot, ul_queue : deque):
        dci0 = list()
        pusch = list()

        # Only schedule in DL slot due to the DCI0
        current_slot = self.env.get_slot_info(frame = frame, slot = slot)
        if current_slot != 'D':
            return dci0, pusch
        
        contention_size = 0
        schedule_size = 0
        cumulated_rb = 0

        while len(ul_queue) > 0 :
            ul_ue : UE = ul_queue.popleft()
            ul_ue.freq_leng = ul_ue.total_data  / (self.env.tbs_per_RB / 8)
            ul_ue.time_length = 14
            ul_ue.start_symbol = 0
            

            if ul_ue.rdb <= 2 * self.env.pattern_p:
                # Schedule
                ul_ue.contention = False
                schedule_size += ul_ue.freq_leng
            else:
                ul_ue.contention = True
                contention_size += ul_ue.freq_leng
            
            if schedule_size + contention_size >= self.env.nrofRB:
                ul_queue.appendleft(ul_ue)
                break

            # Fill DCI0
            dci = DCI_0_0()
            dci.header = ul_ue.id
            dci.UE_id = ul_ue.id
            dci.frequency_domain_assignment = ul_ue.freq_leng
            dci.time_domain_resource_assignment = ul_ue.time_length
            dci.contention = ul_ue.contention
            if ul_ue.contention:
                dci.contention_size = contention_size
            
            dci.fill_payload()
            dci0.append(dci.payload)

            # Fill PUSCH result
            cumulated_rb += ul_ue.freq_leng

        pusch = self.env.PUSCH_Transition(frame = frame, slot = slot+self.env.k2, cumulatied_rb = cumulated_rb)
        return dci0, pusch
    

    def schedule_pdsch(self):
        return None, None
        

    def schedule_pucch(self):
        return None, None


    def first_come_first_serve(users):
        selected_user = users[0]
        users.pop(0)
        return selected_user

    ############################### FCFS ############################## 

    def FCFS(self):
        print("\n\n[ This is the First-Come-First-Serve ]\n\n")

        # To initial the state
        state_tuple, reward, done = self.env.init_RAN_system()
        ul_req = list()
        ul_queue = deque([], maxlen = 65535) # Max number of the UE
        cumulative_reward = reward
        
        while not done:
            '''
            This idea is from OAI
            1.) Schedule PUSCH and DCI0
            2.) Schedule PDSCH and DCI1
            3.) Schedule PUCCH for UCI
            '''
            frame = state_tuple.frame
            slot = state_tuple.slot
            ul_req : list = state_tuple.ul_req
            action = Schedule_Result()    
            
            # Update the queue
            for i in range(len(ul_queue)):
                ul_queue[i].rdb -= 1
                ul_queue[i].queuing_delay += 1
            
            for ul_ue in ul_req:
                ul_queue.append(ul_ue)
                
            # Schedule the DCI0 and UL Data sequentially
            dci0, pusch_result = self.schedule_pusch(frame = frame, slot = slot, ul_queue = ul_queue) 

            # Schedule the DCI1 and DL data Sequentially
            # dci1, pdsch_result = self.schedule_pdsch()

            # Schedule the PUCCH
            # pucch_result = self.schedule_pucch()

            # Process DCI
            # pdcch_result = []

            action.DCCH = dci0
            # action.DSCH = pdsch_result
            # action.UCCH = pucch_result
            action.USCH = pusch_result
            
            
            state_tuple, reward, done = self.env.step(action)
            cumulative_reward += reward
            time.sleep(0.001)

            
    ###################################################################


    ############################# Train ###############################
    def train(self):
        print("testestset")
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
