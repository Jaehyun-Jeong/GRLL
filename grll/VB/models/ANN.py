from typing import Union

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F


class ANN_V1(nn.Module):
    def __init__(self, inputs, outputs):
        super(ANN_V1, self).__init__()

        self.fc1 = nn.Linear(inputs, 256)
        self.fc2 = nn.Linear(256, outputs)

    def forward(self, x):
        state = x

        value = F.relu(self.fc1(state))
        value = self.fc2(value)

        return value


class ANN_V2(nn.Module):
    def __init__(self, inputs, outputs):
        super(ANN_V2, self).__init__()

        self.fc1 = nn.Linear(inputs, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, outputs)

    def forward(self, x):
        state = x

        value = F.relu(self.fc1(state))
        value = F.relu(self.fc2(value))
        value = self.fc3(value)

        return value


class ANN_V3(nn.Module):
    def __init__(self, inputs, outputs):
        super(ANN_V3, self).__init__()

        self.fc1 = nn.Linear(inputs, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, outputs)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        state = x

        value = F.relu(self.fc1(state))
        value = F.relu(self.fc2(value))
        value = self.fc3(value)

        return value


class ANN_V4(nn.Module):
    def __init__(
            self,
            inputs: Union[torch.Tensor, int],
            outputs: Union[torch.Tensor, int]) -> torch.Tensor:

        super(ANN_V4, self).__init__()

        self.fc1 = nn.Linear(inputs, 2*inputs)
        self.fc2 = nn.Linear(2*inputs, 2*inputs)
        self.fc3 = nn.Linear(2*inputs, inputs)
        self.fc4 = nn.Linear(inputs, outputs)

    def forward(self, x):
        state = x

        value = torch.tanh(self.fc1(state))
        value = torch.tanh(self.fc2(value))
        value = torch.tanh(self.fc3(value))
        value = self.fc4(value)

        return value


class ANN_Maze(nn.Module):
    def __init__(
            self,
            inputs: Union[torch.Tensor, int],
            outputs: Union[torch.Tensor, int]) -> torch.Tensor:

        super(ANN_Maze, self).__init__()

        self.layers = nn.Sequential(
                nn.Linear(inputs, inputs*2),
                nn.ReLU(),
                nn.Linear(inputs*2, inputs*2),
                nn.ReLU(),
                nn.Linear(inputs*2, inputs*2),
                nn.ReLU(),
                nn.Linear(inputs*2, inputs*2),
                nn.ReLU(),
                nn.Linear(inputs*2, inputs),
                nn.ReLU(),
                nn.Linear(inputs, outputs),
            )

    def forward(self, x):

        value = self.layers(x)

        return value


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
                nn.Linear(50, outputs),
            )

    def forward(self, x):

        value = self.layers(x)

        return value
