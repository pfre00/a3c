import argparse

import torch
import torch.multiprocessing as mp

import gym

from model import ActorCritic
from train import train
from async_rmsprop import AsyncRMSprop

# Training settings
parser = argparse.ArgumentParser(description='A3C')
parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                    help='learning rate (default: 0.0001)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for rewards (default: 0.99)')
parser.add_argument('--tau', type=float, default=1.00, metavar='L',
                    help='parameter for GAE (default: 1.00)')
parser.add_argument('--num-processes', type=int, default=4, metavar='N',
                    help='how many training processes to use (default: 4)')
parser.add_argument('--num-steps', type=int, default=20, metavar='NS',
                    help='number of forward steps in A3C (default: 20)')
parser.add_argument('--env-name', default='Breakout-v0', metavar='ENV',
                    help='environment to train on (default: Breakout-v0)')


if __name__ == '__main__':
    args = parser.parse_args()
    torch.set_num_threads(1)

    env = gym.make(args.env_name)
    
    global_model = ActorCritic(env.action_space.n)
    global_model.share_memory()
    local_model = ActorCritic(env.action_space.n)
    
    optimizer = AsyncRMSprop(global_model.parameters(), local_model.parameters(), lr=args.lr)

    processes = []
    for rank in range(args.num_processes):
        p = mp.Process(target=train, args=(rank, args, global_model, local_model, optimizer))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
