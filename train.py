import math

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torchvision

import gym

from datetime import datetime, timedelta

convert_state = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Lambda(lambda x: Variable(x.unsqueeze(0))),
])

def train(rank, args, global_model, local_model, optimizer):
    torch.manual_seed(args.seed + rank)
    torch.set_num_threads(1)
    
    t_start = datetime.now()
    
    env = gym.make(args.env_name)
    env.seed(args.seed + rank)
    
    state = env.reset()
    done = True

    reward_sum = 0
    running_reward = 0
    episodes = 0
    
    if rank == 0:
        print("t_now\tt_elapsed\tepisodes\trunning_reward\treward_sum")
    
    while True:
        
        local_model.load_state_dict(global_model.state_dict())
        
        if done:
            cx = Variable(torch.zeros(1, 128))
            hx = Variable(torch.zeros(1, 128))
        else:
            cx = Variable(cx.data)
            hx = Variable(hx.data)

        values = []
        log_probs = []
        rewards = []
        entropies = []

        for step in range(args.num_steps):
            
            if rank == 0 and args.render:
                env.render()
            
            mu, sigma2, value, (hx, cx) = local_model((convert_state(state), (hx, cx)))
            sigma2 = (1 + sigma2.exp()).log()
            
            entropy = (- 0.5 * ((2 * math.pi * sigma2).log() + 1)).sum(1)
            
            action = Variable(torch.normal(mu, sigma2.sqrt()).data)
            log_prob = ((1 / (2 * sigma2 * math.pi).sqrt()).log() - (action - mu) ** 2 / ( 2 * sigma2)).sum(1)
            #log_prob = ( - sigma2.sqrt().log() - (action - mu) ** 2 / ( 2 * sigma2)).sum(1)
            #log_prob = ((action - mu) / sigma2).sum(1)
            
            state, reward, done, _ = env.step(action.data[0])
            
            values.append(value)
            log_probs.append(log_prob)
            rewards.append(max(min(reward, 1), -1))
            entropies.append(entropy)
            
            reward_sum += reward
            
            if done:
                t_now = datetime.now()
                t_elapsed = t_now - t_start
                episodes += 1
                running_reward = running_reward * 0.99 + reward_sum * 0.01
                unbiased_running_reward = running_reward / (1 - pow(0.99, episodes))
                if rank == 0:
                    print("{}\t{}\t{}\t{:.2f}\t{}".format(
                        t_now, t_elapsed, episodes, unbiased_running_reward, reward_sum))
                reward_sum = 0
                state = env.reset()
                break

        R = torch.zeros(1, 1)
        if not done:
            _, _, value, _ = local_model((convert_state(state), (hx, cx)))
            R = value.data
        R = Variable(R)
        
        values.append(R)
        
        gae = torch.zeros(1, 1)
        
        policy_loss = 0
        value_loss = 0
        
        for t in reversed(range(len(rewards))):
            R = args.gamma * R + rewards[t]
            
            advantage = R - values[t]
            value_loss = value_loss + advantage.pow(2)
            
            # Generalized Advantage Estimataion
            delta_t = rewards[t] + args.gamma * values[t + 1].data - values[t].data
            gae = gae * args.gamma * args.tau + delta_t

            policy_loss = policy_loss - log_probs[t] * Variable(gae) - 0.0001 * entropies[t]
            
        optimizer.zero_grad()
        (policy_loss + 0.5 * value_loss).backward()
        optimizer.step()
