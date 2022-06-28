# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F

class ANN_V1(nn.Module):
    def __init__(self, inputs, outputs):
        super(ANN_V1, self).__init__()

        # for Actor
        self.fc1 = nn.Linear(inputs, 256)
        self.actor_fc2 = nn.Linear(256, outputs)
        self.head = nn.Softmax(dim=0)

        # for Critic
        self.critic_fc2 = nn.Linear(256, 1)
        
    def forward(self, x):
        state = x
        state = F.relu(self.fc1(state))

        probs = self.head(self.actor_fc2(state))
        value = self.critic_fc2(state)
        
        return value, probs

class ANN_V2(nn.Module):
    def __init__(self, inputs, outputs):
        super(ANN_V2, self).__init__()

        # for Actor
        self.fc1 = nn.Linear(inputs, 256)
        self.fc2 = nn.Linear(256, 256)
        self.actor_fc3 = nn.Linear(256, outputs)
        self.head = nn.Softmax(dim=0)

        # for Critic
        self.critic_fc3 = nn.Linear(256, 1)
        
    def forward(self, x):
        state = x
        
        state = F.relu(self.fc1(state))
        state = F.relu(self.fc2(state))

        probs = self.head(self.actor_fc3(state))
        value = self.critic_fc3(state)
        
        return value, probs
