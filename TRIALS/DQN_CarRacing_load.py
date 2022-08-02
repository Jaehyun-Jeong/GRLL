import sys
sys.path.append("../") # to import module

# PyTorch
import torch
import torch.optim as optim

# import model
from module.ValueBased.models import CNN_V2
from module.ValueBased import DQN

# Environment
from module.envs.CarRacing import RacingEnv_v2

# set environment
trainEnv = RacingEnv_v2(ExploringStarts=True)
testEnv = RacingEnv_v2()

# set ActorCritic
num_actions = trainEnv.num_actions
num_states = trainEnv.num_obs

model = CNN_V2(num_states, num_actions)
optimizer = optim.Adam(model.parameters(), lr=0.0001)

params_dict = {
    'trainEnv': trainEnv, # environment like gym
    'testEnv': testEnv, # environment like gym
    'model': model, # torch models for policy and value funciton
    'optimizer': optimizer, # torch optimizer
}

# Initialize Actor-Critic Mehtod
DeepQN = DQN(**params_dict)

# load pretrained model
DeepQN.load("./saved_models/CarRacing_v2/DQN_lr1e-3.obj")

# TRAIN Agent
DeepQN.isRender["test"] = True
DeepQN.test()
