import sys
sys.path.append("../../../") # to import module

# PyTorch
import torch
import torch.optim as optim

# import model
from grll.VB.models import ANN_V2
from grll.VB import DQN

# Environment 
import gymnasium as gym

lr = 0.001

env = gym.make('CartPole-v0')
num_actions = env.action_space.n
num_states = env.observation_space.shape[0]

# set ActorCritic
model = ANN_V2(num_states, num_actions)
optimizer = optim.Adam(model.parameters(), lr=lr)

# Initialize Actor-Critic Mehtod
DeepQN = DQN(
    model=model, # torch models for policy and value funciton
    env=env,
    optimizer=optimizer, # torch optimizer
    maxMemory=100000,
    numBatch=32,
)

DQN.isRender = {
    'train': False,
    'test': True,
}

# TRAIN Agent
DeepQN.train(trainTimesteps=1_000_000, testPer=1000)
