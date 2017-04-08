import gym
import numpy as np
import universe
from gym.spaces.box import Box
from universe import vectorized
from universe.wrappers import Unvectorize, Vectorize

import torchvision
from torch.autograd import Variable


# Taken from https://github.com/openai/universe-starter-agent
def create_atari_env(env_id):
    env = gym.make(env_id)
    if len(env.observation_space.shape) > 1:
        env = Vectorize(env)
        env = Process(env)
        env = Normalize(env)
        env = ToVariable(env)
        env = Unvectorize(env)
    return env

_process_frame = torchvision.transforms.Compose([
    torchvision.transforms.ToPILImage(),
    torchvision.transforms.Lambda(lambda x: x.convert('L')),
    torchvision.transforms.Scale(80),
    torchvision.transforms.Scale(42),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Lambda(lambda x: x.unsqueeze(0))
])

class Process(vectorized.ObservationWrapper):

    def __init__(self, env=None):
        super(Process, self).__init__(env)
        self.observation_space = Box(0.0, 1.0, [1, 42, 42])

    def _observation(self, observation_n):
        return [_process_frame(observation) for observation in observation_n]

class Normalize(vectorized.ObservationWrapper):

    def __init__(self, env=None):
        super(Normalize, self).__init__(env)
        self.state_mean = 0
        self.state_std = 0
        self.alpha = 0.9999
        self.num_steps = 0

    def _observation(self, observation_n):
        for observation in observation_n:
            self.num_steps += 1
            self.state_mean = self.state_mean * self.alpha + observation.mean() * (1 - self.alpha)
            self.state_std = self.state_std * self.alpha + observation.std() * (1 - self.alpha)

        unbiased_mean = self.state_mean / (1 - pow(self.alpha, self.num_steps))
        unbiased_std = self.state_std / (1 - pow(self.alpha, self.num_steps))
        
        return [(observation - unbiased_mean) / (unbiased_std + 1e-8) for observation in observation_n]

class ToVariable(vectorized.ObservationWrapper):
    
    def __init__(self, env=None):
        super(ToVariable, self).__init__(env)

    def _observation(self, observation_n):
        return [Variable(observation) for observation in observation_n]