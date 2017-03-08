import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torchvision

import gym

from datetime import datetime, timedelta

convert_state = torchvision.transforms.Compose([
    torchvision.transforms.ToPILImage(),
    #torchvision.transforms.Lambda(lambda x: x.convert('L')),
    #torchvision.transforms.Scale(84),
    torchvision.transforms.ToTensor(),
    #torchvision.transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),
    torchvision.transforms.Lambda(lambda x: x.unsqueeze(0)),
])

def train(rank, args, global_model, local_model, optimizer):
    #torch.manual_seed(args.seed + rank)

    t_start = datetime.now()
    
    env = gym.make(args.env_name)
    #env.seed(args.seed + rank)
    
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
            
            if rank == 0 and args.render:
                env.render()
            
            value, logit, (hx, cx) = local_model((Variable(convert_state(state)), (hx, cx)))
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
            value, _, _ = local_model((Variable(convert_state(state)), (hx, cx)))
            R = value.data
        R = Variable(R)
        
        values.append(R)
        
        gae = torch.zeros(1, 1)
        
        policy_loss = 0
        value_loss = 0
        
        for t in reversed(range(len(rewards))):
            
            R = rewards[t] + args.gamma * R
            advantage = R - values[t]
            
            #policy_loss = policy_loss - log_probs[t] * Variable(advantage.data) - 0.01 * entropies[t]
            
            # Generalized Advantage Estimataion
            delta_t = rewards[t] + args.gamma * values[t + 1].data - values[t].data
            gae = gae * args.gamma * args.tau + delta_t

            policy_loss = policy_loss - log_probs[t] * Variable(gae) - 0.01 * entropies[t]
            
            value_loss = value_loss + advantage.pow(2)
            
        optimizer.zero_grad()
        (policy_loss + 0.5 * value_loss).backward()
        optimizer.step()
