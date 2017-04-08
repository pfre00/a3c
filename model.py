import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

conv_out = 32*4*3
lstm_out = 256

class ActorCritic(nn.Module):

    def __init__(self, num_actions):
        super(ActorCritic, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        
        self.lstm = nn.LSTMCell(conv_out, lstm_out)
        
        self.actor_linear = nn.Linear(lstm_out, num_actions)
        self.critic_linear = nn.Linear(lstm_out, 1)
        
        self.train()
    
    def forward(self, x, t, (hx, cx)):
        x = F.elu(self.conv1(x))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        x = F.elu(self.conv4(x))
        x = x.view(-1, conv_out)
        
        hx, cx = self.lstm(x, (hx, cx))
        x = hx

        return self.critic_linear(x), self.actor_linear(x), (hx, cx)
