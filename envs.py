import gym
import numpy as np
from gym.spaces.box import Box

import torchvision
from torch.autograd import Variable


# Taken from https://github.com/openai/universe-starter-agent
def create_atari_env(env_id):
    env = gym.make(env_id)
    env = Process(env)
    env = Normalize(env)
    env = ToVariable(env)
    return env

_process_frame = torchvision.transforms.Compose([
    torchvision.transforms.ToPILImage(),
    #torchvision.transforms.Lambda(lambda x: x.convert('L')),
    #torchvision.transforms.Scale(80),
    #torchvision.transforms.Scale(42),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Lambda(lambda x: x.unsqueeze(0))
])

class Process(gym.ObservationWrapper):

    def __init__(self, env=None):
        super(Process, self).__init__(env)
        self.observation_space = Box(0.0, 1.0, [3, 160, 210])

    def _observation(self, observation):
        return _process_frame(observation)

class Normalize(gym.ObservationWrapper):

    def __init__(self, env=None):
        super(Normalize, self).__init__(env)
        self.state_mean = 0
        self.state_std = 0
        self.alpha = 0.9999
        self.num_steps = 0

    def _observation(self, observation):
        self.num_steps += 1
        self.state_mean = self.state_mean * self.alpha + observation.mean() * (1 - self.alpha)
        self.state_std = self.state_std * self.alpha + observation.std() * (1 - self.alpha)

        unbiased_mean = self.state_mean / (1 - pow(self.alpha, self.num_steps))
        unbiased_std = self.state_std / (1 - pow(self.alpha, self.num_steps))
        
        return (observation - unbiased_mean) / (unbiased_std or 1)

class ToVariable(gym.ObservationWrapper):
    
    def __init__(self, env=None):
        super(ToVariable, self).__init__(env)

    def _observation(self, observation):
        return Variable(observation)