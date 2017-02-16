import torch.nn as nn
import torch.nn.functional as F

class ActorCritic(nn.Module):

    def __init__(self, num_inputs, action_space):
        super(ActorCritic, self).__init__()
        
        self.conv1 = nn.Conv2d(num_inputs, 16, 8, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 24, 6, stride=4, padding=1)
        self.conv3 = nn.Conv2d(24, 32, 3, stride=3, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        
        self.lstm = nn.LSTMCell(640, 256)
        
        self.actor_linear = nn.Linear(256, action_space)
        self.critic_linear = nn.Linear(256, 1)
        
        self.train()
    
    def forward(self, inputs):
        
        x, (hx, cx) = inputs
        
        x = F.elu(self.conv1(x))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        x = F.elu(self.conv4(x))
        
        x = x.view(-1, 640)
        
        hx, cx = self.lstm(x, (hx, cx))
        x = hx

        return self.critic_linear(x), self.actor_linear(x), (hx, cx)
