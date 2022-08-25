from typing import Union

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F


class ANN_V1(nn.Module):
    def __init__(self, inputs: int, outputs: int):
        super(ANN_V1, self).__init__()

        # for Actor
        self.actor_fc1 = nn.Linear(inputs, 256)
        self.actor_fc2 = nn.Linear(256, outputs)
        self.head = nn.Softmax(dim=0)

        # for Critic
        self.critic_fc1 = nn.Linear(inputs, 256)
        self.critic_fc2 = nn.Linear(256, 1)

    def forward(self, x: torch.Tensor) \
            -> tuple[torch.Tensor, torch.Tensor]:

        state = x

        probs = F.relu(self.actor_fc1(state))
        probs = self.head(self.actor_fc2(probs))

        value = F.relu(self.critic_fc1(state))
        value = self.critic_fc2(value)

        return value, probs


class ANN_V2(nn.Module):
    def __init__(self, inputs: int, outputs: int):
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

    def forward(self, x: torch.Tensor) \
            -> tuple[torch.Tensor, torch.Tensor]:

        state = x

        probs = F.relu(self.actor_fc1(state))
        probs = F.relu(self.actor_fc2(probs))
        probs = self.head(self.actor_fc3(probs))

        value = F.relu(self.critic_fc1(state))
        value = F.relu(self.critic_fc2(value))
        value = self.critic_fc3(value)

        return value, probs


class ANN_V3(nn.Module):
    def __init__(
            self,
            inputs: Union[torch.Tensor, int],
            outputs: Union[torch.Tensor, int]) -> torch.Tensor:

        super(ANN_V3, self).__init__()

        # Actor
        self.actor_fc1 = nn.Linear(inputs, 64)
        self.actor_fc2 = nn.Linear(64, 64)
        self.actor_fc3 = nn.Linear(64, outputs)
        self.head = nn.Softmax(dim=0)

        # Critic
        self.critic_fc1 = nn.Linear(inputs, 64)
        self.critic_fc2 = nn.Linear(64, 64)
        self.critic_fc3 = nn.Linear(64, outputs)

    def forward(self, x):
        state = x

        probs = F.tanh(self.actor_fc1(state))
        probs = F.tanh(self.actor_fc2(probs))
        probs = self.head(self.actor_fc3(probs))

        value = F.tanh(self.critic_fc1(state))
        value = F.tanh(self.critic_fc2(value))
        value = self.critic_fc3(value)

        return value, probs
