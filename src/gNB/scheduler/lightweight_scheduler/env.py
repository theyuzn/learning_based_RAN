from ue.ue import UE

MAX_GROUP = 3 + 1 # +1 = scheduled group


class observation(object):
        nrofUE = 0
        def __init__(self, UE):
            self.nrofUE = 0


class Environment(object):
    def __init__(self, uelist : list(UE), seed = 0):
         self.uelist = uelist
         self.seed = seed
         
        # self.game = gym.make(game)
        # self.game.seed(seed)

        # if record:
        #     self.game = Monitor(self.game, './video', force=True)

        # self.width = width
        # self.height = height
        # self._toTensor = T.Compose([T.ToPILImage(), T.ToTensor()])
        # gym_ple

    # def play_sample(self, mode: str = 'human'):
    #     observation = self.game.reset()

    #     while True:
    #         screen = self.game.render(mode=mode)
    #         if mode == 'rgb_array':
    #             screen = self.preprocess(screen)
    #         action = self.game.action_space.sample()
    #         observation, reward, done, info = self.game.step(action)
    #         if done:
    #             break
    #     self.game.close()

    # def preprocess(self, screen):
    #     preprocessed: np.array = cv2.resize(screen, (self.height, self.width))  
    #     preprocessed = np.dot(preprocessed[..., :3], [0.299, 0.587, 0.114]) 
    #     # preprocessed: np.array = preprocessed.transpose((2, 0, 1)) 
    #     preprocessed: np.array = preprocessed.astype('float32') / 255.

    #     return preprocessed

    # def init(self):
    #     """
    #     @return observation
    #     """
    #     return self.game.reset()

    # def get_screen(self):
    #     screen = self.game.render('rgb_array')
    #     screen = self.preprocess(screen)
    #     return screen

    # def step(self, action: int):
    #     observation, reward, done, info = self.game.step(action)
    #     return observation, reward, done, info

    # def reset(self):
    #     """
    #     :return: observation array
    #     """
    #     observation = self.game.reset()
    #     observation = self.preprocess(observation)
    #     return observation
