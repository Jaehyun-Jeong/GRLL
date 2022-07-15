# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F

class ANN_V1(nn.Module):
    def __init__(self, inputs, outputs):
        super(ANN_V1, self).__init__()

        # for Actor
        self.actor_fc1 = nn.Linear(inputs, 256)
        self.actor_fc2 = nn.Linear(256, outputs)
        self.head = nn.Softmax(dim=0)

        # for Critic
        self.critic_fc1 = nn.Linear(inputs, 256)
        self.critic_fc2 = nn.Linear(256, 1)
        
    def forward(self, x):
        
        state = x.to(torch.float32)
        
        probs = F.relu(self.actor_fc1(state))
        probs = self.head(self.actor_fc2(probs))

        value = F.relu(self.critic_fc1(state))
        value = self.critic_fc2(value)
        
        return value, probs

class ANN_V2(nn.Module):
    def __init__(self, inputs, outputs):
        super(ANN_V2, self).__init__()

        # for Actor
        self.actor_fc1 = nn.Linear(inputs, 256)
        self.actor_fc2 = nn.Linear(256, 256)
        self.actor_fc3 = nn.Linear(256, outputs)
        self.head = nn.Softmax(dim=0)

        # for Critic
        self.critic_fc1 = nn.Linear(inputs, 256)
        self.critic_fc2 = nn.Linear(256, 256)
        self.critic_fc3 = nn.Linear(256, 1)
        
    def forward(self, x):
        state = x

        probs = F.relu(self.actor_fc1(state))
        probs = F.relu(self.actor_fc2(probs))
        probs = self.head(self.actor_fc3(probs))
        
        value = F.relu(self.critic_fc1(state))
        value = F.relu(self.critic_fc2(value))
        value = self.critic_fc3(value)
        
        return value, probs
