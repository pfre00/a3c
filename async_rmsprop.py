from torch.optim import Optimizer

class AsyncRMSprop(Optimizer):
    def __init__(self, global_params, local_params, lr=1e-2, alpha=0.99, eps=0.1, weight_decay=0):
        defaults = dict(lr=lr, alpha=alpha, eps=eps, weight_decay=weight_decay)
        super(AsyncRMSprop, self).__init__(global_params, defaults)

        self.local_param_groups = list(local_params)
        if not isinstance(self.local_param_groups[0], dict):
            self.local_param_groups = [{'params': self.local_param_groups}]
        
        # State initialization
        for l_group, group in zip(self.local_param_groups, self.param_groups):
            for l_p, p in zip(l_group['params'], group['params']):
                grad = l_p.grad.data
                state = self.state[p]
                state['step'] = 0
                state['square_avg'] = grad.new().resize_as_(grad).zero_().share_memory_()

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for l_group, group in zip(self.local_param_groups, self.param_groups):
            for l_p, p in zip(l_group['params'], group['params']):
                grad = l_p.grad.data
                state = self.state[p]
                
                square_avg = state['square_avg']
                alpha = group['alpha']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)

                square_avg.mul_(alpha).addcmul_(1 - alpha, grad, grad)
                avg = square_avg.sqrt().add_(group['eps'])
                p.data.addcdiv_(-group['lr'], grad, avg)

        return loss

    def zero_grad(self):
        for l_group in self.local_param_groups:
            for l_p in l_group['params']:
                l_p.grad.data.zero_()