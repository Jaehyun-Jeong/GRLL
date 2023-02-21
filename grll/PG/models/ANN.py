from typing import Union, Tuple

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

        # for Critic
        self.critic_fc1 = nn.Linear(inputs, 256)
        self.critic_fc2 = nn.Linear(256, 1)

    def forward(self, x: torch.Tensor) \
            -> tuple[torch.Tensor, torch.Tensor]:

        state = x

        probs = F.relu(self.actor_fc1(state))
        probs = self.actor_fc2(probs)

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

        # for Critic
        self.critic_fc1 = nn.Linear(inputs, 256)
        self.critic_fc2 = nn.Linear(256, 256)
        self.critic_fc3 = nn.Linear(256, 1)

    def forward(self, x: torch.Tensor) \
            -> tuple[torch.Tensor, torch.Tensor]:

        state = x

        probs = F.relu(self.actor_fc1(state))
        probs = F.relu(self.actor_fc2(probs))
        probs = self.actor_fc3(probs)

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

        # Critic
        self.critic_fc1 = nn.Linear(inputs, 64)
        self.critic_fc2 = nn.Linear(64, 64)
        self.critic_fc3 = nn.Linear(64, 1)

    def forward(self, x):
        state = x

        probs = F.relu(self.actor_fc1(state))
        probs = F.relu(self.actor_fc2(probs))
        probs = self.actor_fc3(probs)

        value = F.relu(self.critic_fc1(state))
        value = F.relu(self.critic_fc2(value))
        value = self.critic_fc3(value)

        return value, probs


class ANN_V4(nn.Module):
    def __init__(
            self,
            inputs: Union[torch.Tensor, int],
            outputs: Union[torch.Tensor, int]):

        super(ANN_V4, self).__init__()

        # Actor
        self.actor_fc1 = nn.Linear(inputs, 2*inputs)
        self.actor_fc2 = nn.Linear(2*inputs, 2*inputs)
        self.actor_fc3 = nn.Linear(2*inputs, inputs)
        self.actor_fc4 = nn.Linear(inputs, outputs)

        # Critic
        self.critic_fc1 = nn.Linear(inputs, 2*inputs)
        self.critic_fc2 = nn.Linear(2*inputs, 2*inputs)
        self.critic_fc3 = nn.Linear(2*inputs, inputs)
        self.critic_fc4 = nn.Linear(inputs, 1)

    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        state = x

        probs = torch.tanh(self.actor_fc1(state))
        probs = torch.tanh(self.actor_fc2(probs))
        probs = torch.tanh(self.actor_fc3(probs))
        probs = self.actor_fc4(probs)

        value = torch.tanh(self.critic_fc1(state))
        value = torch.tanh(self.critic_fc2(value))
        value = torch.tanh(self.critic_fc3(value))
        value = self.critic_fc4(value)

        return value, probs


class ANN_V4_shared(nn.Module):
    def __init__(
            self,
            inputs: Union[torch.Tensor, int],
            outputs: Union[torch.Tensor, int]):

        super(ANN_V4, self).__init__()

        self.fc1 = nn.Linear(inputs, 2*inputs)
        self.fc2 = nn.Linear(2*inputs, 2*inputs)
        self.fc3 = nn.Linear(2*inputs, inputs)

        # Actor
        self.actor_fc4 = nn.Linear(inputs, outputs)

        # Critic
        self.critic_fc4 = nn.Linear(inputs, 1)

    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        state = x

        state = torch.tanh(self.fc1(state))
        state = torch.tanh(self.fc2(probs))
        state = torch.tanh(self.fc3(probs))

        probs = self.actor_fc4(probs)
        value = self.critic_fc4(value)

        return value, probs


class ANN_Cal(nn.Module):
    def __init__(
            self,
            inputs: Union[torch.Tensor, int],
            outputs: Union[torch.Tensor, int]) -> torch.Tensor:

        super(ANN_Cal, self).__init__()

        self.layers = nn.Sequential(
                nn.Linear(inputs, 10),
                nn.LeakyReLU(),
                nn.BatchNorm1d(10),
                nn.Linear(10, 20),
                nn.LeakyReLU(),
                nn.BatchNorm1d(20),
                nn.Linear(20, 50),
                nn.LeakyReLU(),
                nn.BatchNorm1d(50),
            )

        self.actor_fc = nn.Sequential(
                nn.Linear(50, outputs),
                nn.LogSoftmax(dim=-1)
            )

        self.critic_fc = nn.Linear(50, 1)

    def forward(self, x):

        x = self.layers(x)
        probs = self.actor_fc(x)
        value = self.critic_fc(x)

        return value, probs
