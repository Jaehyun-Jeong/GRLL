# PyTorch
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

    def forward(self, x):
        state = x

        value = F.relu(self.fc1(state))
        value = F.relu(self.fc2(value))
        value = self.fc3(value)
        
        return value
