import torch.nn as nn
import torch.nn.functional as F

class ActorCritic(nn.Module):

    def __init__(self, action_space):
        super(ActorCritic, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 4, 8, stride=4, padding=3)
        self.conv2 = nn.Conv2d(4, 8, 8, stride=4, padding=3)
        self.conv3 = nn.Conv2d(8, 16, 6, stride=3, padding=2)
        
        self.lstm = nn.LSTMCell(192, 256)
        
        self.actor_linear = nn.Linear(256, action_space)
        self.critic_linear = nn.Linear(256, 1)
        
        self.train()
    
    def forward(self, inputs):
        
        x, (hx, cx) = inputs
        
        x = F.elu(self.conv1(x))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        
        x = x.view(-1, 192)
        
        hx, cx = self.lstm(x, (hx, cx))
        x = hx

        return self.critic_linear(x), self.actor_linear(x), (hx, cx)
