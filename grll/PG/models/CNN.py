# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN_V1(nn.Module):
    def __init__(self, h: int, w: int, outputs: int):

        super(CNN_V1, self).__init__()
        self.actor_conv1 = nn.Conv2d(5, 16, kernel_size=3, stride=1)
        self.actor_conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1)
        self.actor_conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1)
        
        self.critic_conv1 = nn.Conv2d(5, 16, kernel_size=3, stride=1)
        self.critic_conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1)
        self.critic_conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size: int = 3, stride: int = 1):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        
        self.actor_fc1 = nn.Linear(linear_input_size, outputs)
        self.critic_fc1 = nn.Linear(linear_input_size, 1)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x: torch.Tensor) \
            -> tuple[torch.Tensor, torch.Tensor]:

        state = x
        state = torch.unsqueeze(state, 0)
        
        probs = F.relu(self.actor_conv1(state))
        probs = F.relu(self.actor_conv2(probs))
        probs = F.relu(self.actor_conv3(probs))
        probs = torch.flatten(probs, 1)
        probs = self.actor_fc1(probs)
        
        value = F.relu(self.critic_conv1(state))
        value = F.relu(self.critic_conv2(value))
        value = F.relu(self.critic_conv3(value))
        value = torch.flatten(value, 1)
        value = F.relu(self.critic_fc1(value))
        
        return value, probs


class CNN_V2(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        c, h, w = input_dim

        if h != 84:
            raise ValueError(f"Expecting input height: 84, got: {h}")
        if w != 84:
            raise ValueError(f"Expecting input width: 84, got: {w}")

        # This parts are shared
        self.actor_conv1 = nn.Conv2d(in_channels=c, out_channels=32, kernel_size=8, stride=4)
        self.actor_conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.actor_conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        self.actor_linear1 = nn.Linear(3136, 512)
        self.actor_linear2 = nn.Linear(512, output_dim)

        self.critic_conv1 = nn.Conv2d(in_channels=c, out_channels=32, kernel_size=8, stride=4)
        self.critic_conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.critic_conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        self.critic_linear1 = nn.Linear(3136, 512)
        self.critic_linear2 = nn.Linear(512, 1)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x: torch.Tensor) \
            -> tuple[torch.Tensor, torch.Tensor]:
        state = x

        probs = F.relu(self.actor_conv1(state))
        probs = F.relu(self.actor_conv2(probs))
        probs = F.relu(self.actor_conv3(probs))
        probs = torch.flatten(probs, 1)
        probs = F.relu(self.actor_linear1(probs))
        probs = self.actor_linear2(probs)

        value = F.relu(self.critic_conv1(state))
        value = F.relu(self.critic_conv2(value))
        value = F.relu(self.critic_conv3(value))
        value = torch.flatten(value, 1)
        value = F.relu(self.critic_linear1(value))
        value = self.critic_linear2(value)

        return value, probs

class CNN_V2_shared(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        c, h, w = input_dim

        if h != 84:
            raise ValueError(f"Expecting input height: 84, got: {h}")
        if w != 84:
            raise ValueError(f"Expecting input width: 84, got: {w}")

        # This parts are shared
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        self.linear1 = nn.Linear(3136, 512)

        self.actor_linear2 = nn.Linear(512, output_dim)
        self.critic_linear2 = nn.Linear(512, 1)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x: torch.Tensor) \
            -> tuple[torch.Tensor, torch.Tensor]:
        state = x

        state = F.relu(self.conv1(state))
        state = F.relu(self.conv2(state))
        state = F.relu(self.conv3(state))
        state = torch.flatten(state, 1)
        state = F.relu(self.linear1(state))

        probs = self.actor_linear2(state)
        value = self.critic_linear2(state)

        return value, probs


class CNN_V3(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        c, h, w = input_dim

        if h != 84:
            raise ValueError(f"Expecting input height: 84, got: {h}")
        if w != 84:
            raise ValueError(f"Expecting input width: 84, got: {w}")

        # This parts are shared
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        self.linear1 = nn.Linear(3136, 512)

        self.actor_linear2 = nn.Linear(512, output_dim)
        self.critic_linear2 = nn.Linear(512, 1)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x: torch.Tensor) \
            -> tuple[torch.Tensor, torch.Tensor]:
        state = x

        state = F.relu(self.conv1(state))
        state = F.relu(self.conv2(state))
        state = F.relu(self.conv3(state))
        state = torch.flatten(state, 1)
        state = F.relu(self.linear1(state))

        probs = self.actor_linear2(state)
        value = self.critic_linear2(state)

        return value, probs


class CNN_V4(nn.Module):

    def __init__(self, input_dim, output_dim):
        super().__init__()

        c, h, w = input_dim

        # This parts are shared
        self.conv1_1 = nn.Conv2d(
                in_channels=c, out_channels=128, kernel_size=3, stride=1)
        self.conv1_2 = nn.Conv2d(
                in_channels=128, out_channels=6, kernel_size=3, stride=1)
        self.maxpool1 = nn.MaxPool2d((15, 5), stride=(1, 1))
        self.avgpool1 = nn.AvgPool2d((15, 5), stride=(1, 1))

        self.conv2_1 = nn.Conv2d(
                in_channels=c, out_channels=128, kernel_size=3, stride=1)
        self.conv2_2 = nn.Conv2d(
                in_channels=128, out_channels=4, kernel_size=3, stride=1)
        self.maxpool2 = nn.MaxPool2d((15, 5), stride=(1, 1))
        self.avgpool2 = nn.AvgPool2d((15, 5), stride=(1, 1))

        self.linear1 = nn.Linear(160, 128)
        self.linear2 = nn.Linear(128, 64)
        self.actor_linear3 = nn.Linear(64, output_dim)
        self.critic_linear3 = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) \
            -> tuple[torch.Tensor, torch.Tensor]:

        state = x

        a = F.relu(self.conv1_1(state))
        a = F.relu(self.conv1_2(a))
        a1 = self.maxpool1(a)
        a2 = self.avgpool1(a)
        a = torch.cat((a1, a2), dim=1)

        b = F.relu(self.conv2_1(state))
        b = F.relu(self.conv2_2(b))
        b1 = self.maxpool2(b)
        b2 = self.avgpool2(b)
        b = torch.cat((b1, b2), dim=1)

        x = torch.cat((a, b), dim=1)
        x = torch.flatten(x, 1)
        x = self.linear1(x)
        x = self.linear2(x)

        probs = self.actor_linear3(x)
        value = self.critic_linear3(x)

        return value, probs
