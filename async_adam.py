import math

import torch
from torch.optim import Optimizer

class AsyncAdam(Optimizer):
    def __init__(self, global_params, local_params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(AsyncAdam, self).__init__(global_params, defaults)
        
        self.local_param_groups = list(local_params)
        if not isinstance(self.local_param_groups[0], dict):
            self.local_param_groups = [{'params': self.local_param_groups}]
        
        # State initialization
        for l_group, group in zip(self.local_param_groups, self.param_groups):
            for l_p, p in zip(l_group['params'], group['params']):
                if l_p.requires_grad:
                    state = self.state[p]
                    if len(state) == 0:
                        state['step'] = 0
                        # Exponential moving average of gradient values
                        state['exp_avg'] = p.data.new().resize_as_(p.data).zero_()
                        # Exponential moving average of squared gradient values
                        state['exp_avg_sq'] = p.data.new().resize_as_(p.data).zero_()
        
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for l_group, group in zip(self.local_param_groups, self.param_groups):
            for l_p, p in zip(l_group['params'], group['params']):
                if l_p.grad is None:
                    continue
                grad = l_p.grad.data
                state = self.state[p]

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                p.data.addcdiv_(-step_size, exp_avg, denom)

        return loss
    
    def zero_grad(self):
        for l_group in self.local_param_groups:
            for l_p in l_group['params']:
                if l_p.grad is not None:
                    l_p.grad.data.zero_()