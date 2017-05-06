import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

conv_out = 192
lstm_out = 256

class ActorCritic(nn.Module):

    def __init__(self, num_actions):
        super(ActorCritic, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 4, 8, stride=4, padding=3)
        self.conv2 = nn.Conv2d(4, 8, 8, stride=4, padding=3)
        self.conv3 = nn.Conv2d(8, 16, 6, stride=3, padding=2)
        
        init.kaiming_normal(self.conv1.weight.data)
        self.conv1.bias.data.fill_(0)
        init.kaiming_normal(self.conv2.weight.data)
        self.conv2.bias.data.fill_(0)
        init.kaiming_normal(self.conv3.weight.data)
        self.conv3.bias.data.fill_(0)
        
        self.lstm = nn.LSTMCell(conv_out, lstm_out)
        
        self.actor_linear = nn.Linear(lstm_out, num_actions)
        self.critic_linear = nn.Linear(lstm_out, 1)
        
        self.train()
    
    def forward(self, x, (hx, cx)):
        
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, conv_out)
        
        hx, cx = self.lstm(x, (hx, cx))
        x = hx
        
        return self.critic_linear(x), self.actor_linear(x), (hx, cx)
