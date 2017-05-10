import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import gym

from datetime import datetime, timedelta

from model import ActorCritic
from envs import create_atari_env

from visdom import Visdom

viz = Visdom()

def train(rank, args, model, optimizer):
    torch.manual_seed(args.seed + rank)
    
    env = create_atari_env(args.env_name)
    env.seed(args.seed + rank)
    
    running_reward = 0
    episodes = 0
    
    if rank == 0:
        print("t_elapsed\tepisodes\trunning_reward\treward_sum")
    
    t_start = datetime.now()
    
    done = True
    while True:
        
        if done:
            cx = Variable(torch.zeros(1, 256))
            hx = Variable(torch.zeros(1, 256))
            state = env.reset()
            episodes += 1
            reward_sum = 0
        else:
            cx = cx.detach()
            hx = hx.detach()
        
        log_probs = []
        values = []
        rewards = []
        entropies = []
        
        for step in range(args.num_steps):
            
            log_dist, value, (hx, cx) = model(state, (hx, cx))
            dist = log_dist.exp()

            action = dist.multinomial().detach()
            state, reward, done, _ = env.step(action.data[0][0])
            
            log_prob = log_dist.gather(1, action)
            clipped_reward = max(min(reward, 1), -1)
            entropy = -(dist * log_dist).sum(1)
            
            log_probs.append(log_prob)
            values.append(value)
            rewards.append(clipped_reward)
            entropies.append(entropy)
            
            reward_sum += reward
            
            if done:
                torch.save(model.state_dict(), "state_dict.data")
                t_elapsed = datetime.now() - t_start
                running_reward = running_reward * 0.99 + reward_sum * 0.01
                unbiased_running_reward = running_reward / (1 - pow(0.99, episodes))
                if rank == 0:
                    print("{}\t{}\t{:.2f}\t{}".format(
                        t_elapsed, episodes, unbiased_running_reward, reward_sum))
                break

        R = torch.zeros(1, 1)
        if not done:
            _, value, _ = model(state, (hx, cx))
            R = value.data
        R = Variable(R)
        
        values.append(R)
        
        gae = 0
        
        policy_loss = 0
        value_loss = 0
        
        for t in reversed(range(len(rewards))):
            
            # Value estimation
            R = rewards[t] + args.gamma * R
            
            # Generalized Advantage Estimataion
            td_error = rewards[t] + args.gamma * values[t+1].data - values[t].data
            gae = td_error + args.gamma * args.tau * gae
            
            policy_loss = policy_loss - (log_probs[t] * Variable(gae) + 0.01 * entropies[t])
            
            value_loss = value_loss + (R - values[t])**2
            
        optimizer.zero_grad()
        (policy_loss + 0.5 * value_loss).backward()
        #torch.nn.utils.clip_grad_norm(model.parameters(), 40)
        optimizer.step()