import torch.nn as nn
import torch.nn.functional as F

conv_out = 1120
lstm_out = 256

class ActorCritic(nn.Module):

    def __init__(self, num_actions):
        super(ActorCritic, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 16, 8, stride=4, padding=3)
        self.conv2 = nn.Conv2d(16, 32, 4, stride=2, padding=1)
        
        self.lstm = nn.LSTMCell(conv_out, lstm_out)
        
        self.actor_linear = nn.Linear(lstm_out, num_actions)
        self.critic_linear = nn.Linear(lstm_out, 1)
        
        self.train()
    
    def forward(self, inputs):
        
        x, (hx, cx) = inputs
        
        x = F.elu(self.conv1(x))
        x = F.elu(self.conv2(x))
        x = x.view(-1, conv_out)
        
        hx, cx = self.lstm(x, (hx, cx))
        x = hx

        return self.critic_linear(x), self.actor_linear(x), (hx, cx)
