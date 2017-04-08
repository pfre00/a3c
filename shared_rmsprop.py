import torch
import torch.optim as optim


class SharedRMSprop(optim.RMSprop):
    
    def __init__(self, params, lr=1e-2, alpha=0.99, eps=1e-8, weight_decay=0, momentum=0, centered=False):
        super(SharedRMSprop, self).__init__(params, lr, alpha, eps, weight_decay, momentum, centered)
        
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = torch.zeros(1)
                state['square_avg'] = p.data.new().resize_as_(p.data).zero_()
                if group['momentum'] > 0:
                    state['momentum_buffer'] = p.data.new().resize_as_(p.data).zero_()
                if group['centered']:
                    state['grad_avg'] = p.data.new().resize_as_(p.data).zero_()
    
    def share_memory(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'].share_memory_()
                state['square_avg'].share_memory_()
                if group['momentum'] > 0:
                    state['momentum_buffer'].share_memory_()
                if group['centered']:
                    state['grad_avg'].share_memory_()