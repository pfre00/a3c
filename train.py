import math

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import gym

from model import ActorCritic

def convert_state(state):
    return torch.from_numpy(state).float().permute(2, 0, 1).unsqueeze(0)

def train(rank, args, model):

    env = gym.make(args.env_name)

    for param in model.parameters():
        # Break gradient sharing
        param.grad.data = param.grad.data.clone()

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    state = env.reset()
    reward_sum = 0
    done = True

    running_reward = 0
    num_updates = 0
    
    while True:
        
        if done:
            cx = Variable(torch.zeros(1, 256))
            hx = Variable(torch.zeros(1, 256))
        else:
            cx = Variable(cx.data)
            hx = Variable(hx.data)

        values = []
        log_probs = []
        rewards = []
        entropies = []

        for step in range(args.num_steps):
            
            if rank == 0:
                env.render()
            
            value, logit, (hx, cx) = model((Variable(convert_state(state)), (hx, cx)))
            prob = F.softmax(logit)
            log_prob = F.log_softmax(logit)
            entropy = -(log_prob * prob).sum(1)

            action = Variable(prob.multinomial().data)
            log_prob = log_prob.gather(1, action)

            state, reward, done, _ = env.step(action.data[0][0])
            
            values.append(value)
            log_probs.append(log_prob)
            rewards.append(max(min(reward, 1), -1))
            entropies.append(entropy)
            
            reward_sum += reward
            
            if done:
                running_reward = running_reward * 0.9 + reward_sum * 0.1
                num_updates += 1

                if rank == 0:
                    print("Agent {2}, episodes {0}, running reward {1:.2f}, current reward {3}".format(
                        num_updates, running_reward / (1 - pow(0.9, num_updates)), rank, reward_sum))
                reward_sum = 0
                state = env.reset()
                break

        R = torch.zeros(1, 1)
        if not done:
            value, _, _ = model((Variable(convert_state(state)), (hx, cx)))
            R = value.data
        R = Variable(R)
        
        policy_loss = 0
        value_loss = 0
        
        for t in reversed(range(len(rewards))):
            
            R_t = args.gamma * R + rewards[t]
            V_t = values[t]
            A_t = R_t - V_t
            log_prob_t = log_probs[t]
            H = entropies[t]
            
            value_loss = value_loss + A_t.pow(2)
            policy_loss = policy_loss - log_prob_t * A_t - 0.01 * H

        optimizer.zero_grad()
        (policy_loss + 0.25 * value_loss).backward()
        
        '''
        global_norm = 0
        for param in model.parameters():
            global_norm += param.grad.data.pow(2).sum()
        global_norm = math.sqrt(global_norm)
        ratio = 10 / global_norm
        if ratio < 1:
            for param in model.parameters():
                param.grad.data.mul_(ratio)
        '''
        
        optimizer.step()
