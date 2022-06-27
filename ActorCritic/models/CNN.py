# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN_V1(nn.Module):
    def __init__(self, h, w, outputs):
        super(CNN_V1, self).__init__()
        self.actor_conv1 = nn.Conv2d(5, 16, kernel_size=3, stride=1)
        self.actor_maxPool1 = nn.MaxPool2d(2, 2)
        self.actor_conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1)
        self.actor_maxPool2 = nn.MaxPool2d(2, 2)
        self.actor_conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1)
        
        self.critic_conv1 = nn.Conv2d(5, 16, kernel_size=3, stride=1)
        self.critic_maxPool1 = nn.MaxPool2d(2, 2)
        self.critic_conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1)
        self.critic_maxPool2 = nn.MaxPool2d(2, 2)
        self.critic_conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1)
        
        # torch.log makes nan(not a number) error so we have to add some small number in log function
        self.ups=1e-7

        def conv2d_size_out(size, kernel_size = 3, stride = 1):
            return (size - (kernel_size - 1) - 1) // stride  + 1
         

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        
        self.actor_fc1 = nn.Linear(32, outputs)
        self.head = nn.Softmax(dim=1)
        
        self.critic_fc1 = nn.Linear(32, 1)

    def forward(self, x):
        state = x
        state = torch.unsqueeze(state, 0)
        
        probs = self.actor_maxPool1(self.actor_conv1(state))
        probs = self.actor_maxPool2(self.actor_conv2(probs))
        probs = self.actor_conv3(probs)
        probs = torch.flatten(probs, 1)
        probs = self.head(self.actor_fc1(probs))
        
        value = self.critic_maxPool1(self.critic_conv1(state))
        value = self.critic_maxPool2(self.critic_conv2(value))
        value = self.critic_conv3(value)
        value = torch.flatten(value, 1)
        value = F.relu(self.critic_fc1(value))
        
        return value, probs
