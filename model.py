import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from batch_renorm import BatchRenorm

ff_out = 200
lstm_out = 128

class ActorCritic(nn.Module):

    def __init__(self, num_inputs, num_actions):
        super(ActorCritic, self).__init__()
        
        self.lin1 = nn.Linear(num_inputs, ff_out)
        
        self.lstm = nn.LSTMCell(ff_out, lstm_out)
        
        self.mu = nn.Linear(lstm_out, num_actions)
        self.sigma2 = nn.Linear(lstm_out, num_actions)
        self.value = nn.Linear(lstm_out, 1)
        
        self.train()
    
    def forward(self, inputs):
        
        x, (hx, cx) = inputs
        
        x = F.elu(self.lin1(x))
        
        hx, cx = self.lstm(x, (hx, cx))
        x = hx

        return self.mu(x), self.sigma2(x), self.value(x), (hx, cx)
