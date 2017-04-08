import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import gym

from datetime import datetime, timedelta

from model import ActorCritic
from envs import create_atari_env

def ensure_shared_grads(model, shared_model):
    for param, shared_param in zip(model.parameters(), shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad

def train(rank, args, shared_model, optimizer):
    torch.manual_seed(args.seed + rank)
    torch.set_num_threads(1)
    
    t_start = datetime.now()
    
    env = create_atari_env(args.env_name)
    env.seed(args.seed + rank)
    
    model = ActorCritic(env.action_space.n)
    
    running_reward = 0
    episodes = 0
    
    if rank == 0:
        print("t_elapsed\tepisodes\trunning_reward\treward_sum")
    
    done = True
    
    while True:
        
        model.load_state_dict(shared_model.state_dict())
        
        if done:
            cx = Variable(torch.zeros(1, 256))
            hx = Variable(torch.zeros(1, 256))
            state = env.reset()
            episodes += 1
            game_step = 0
            reward_sum = 0
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
            
            value, logit, (hx, cx) = model(state, game_step, (hx, cx))
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
            game_step += 1
            
            if done:
                t_now = datetime.now()
                t_elapsed = t_now - t_start
                running_reward = running_reward * 0.99 + reward_sum * 0.01
                unbiased_running_reward = running_reward / (1 - pow(0.99, episodes))
                if rank == 0:
                    print("{}\t{}\t{:.2f}\t{}".format(
                        t_elapsed, episodes, unbiased_running_reward, reward_sum))
                break

        R = torch.zeros(1, 1)
        if not done:
            value, _, _ = model(state, game_step, (hx, cx))
            R = value.data
        R = Variable(R)
        
        values.append(R)
        
        gae = torch.zeros(1, 1)
        
        policy_loss = 0
        value_loss = 0
        
        for t in reversed(range(len(rewards))):
            
            R = rewards[t] + args.gamma * R
            advantage = R - values[t]
            
            value_loss = value_loss + advantage.pow(2)
            
            # Generalized Advantage Estimataion
            delta_t = rewards[t] + args.gamma * values[t + 1].data - values[t].data
            gae = gae * args.gamma * args.tau + delta_t

            policy_loss = policy_loss - log_probs[t] * Variable(gae) - 0.01 * entropies[t]
            
        optimizer.zero_grad()
        (policy_loss + 0.5 * value_loss).backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), 40)
        ensure_shared_grads(model, shared_model)
        optimizer.step()