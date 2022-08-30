# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN_V2(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        c, h, w = input_dim

        if h != 84:
            raise ValueError(f"Expecting input height: 84, got: {h}")
        if w != 84:
            raise ValueError(f"Expecting input width: 84, got: {w}")

        self.conv1 = nn.Conv2d(in_channels=c, out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        self.linear1 = nn.Linear(3136, 512)
        self.linear2 = nn.Linear(512, output_dim)
        
    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        state = x

        value = F.relu(self.conv1(state))
        value = F.relu(self.conv2(value))
        value = F.relu(self.conv3(value))
        value = torch.flatten(value, 1)
        value = F.relu(self.linear1(value))
        value = self.linear2(value)
        
        return value
