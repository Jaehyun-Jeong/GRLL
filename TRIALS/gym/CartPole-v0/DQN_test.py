import sys
sys.path.append("../../../") # to import module

# PyTorch
import torch
import torch.optim as optim

# import model
from grll.VB.models import ANN_V2
from grll.VB import DQN

# Environment 
import gym

env = gym.make('CartPole-v0')
num_actions = env.action_space.n
num_states = env.observation_space.shape[0]

# set ActorCritic
model = ANN_V2(num_states, num_actions)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Initialize Actor-Critic Mehtod
DeepQN = DQN(
    model=model, # torch models for policy and value funciton
    env=env,
    optimizer=optimizer, # torch optimizer
    maxMemory=100000,
    numBatch=32,
    verbose=1,
)

# TRAIN Agent
DeepQN.train(trainTimesteps=70000, testPer=10000)
