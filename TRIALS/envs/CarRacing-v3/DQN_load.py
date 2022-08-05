import sys
sys.path.append("../../../") # to import module

# PyTorch
import torch
import torch.optim as optim

# import model
from module.ValueBased.models import CNN_V2
from module.ValueBased import DQN

# Environment
from module.envs.CarRacing import RacingEnv_v3

# set environment
trainEnv = RacingEnv_v3(ExploringStarts=True)
testEnv = RacingEnv_v3()
num_actions = trainEnv.num_actions
num_states = trainEnv.num_obs

model = CNN_V2(num_states, num_actions)
optimizer = optim.Adam(model.parameters(), lr=0.0001)

params_dict = {
    'trainEnv': trainEnv, # environment like gym
    'testEnv': testEnv, # environment like gym
    'model': model, # torch models for policy and value funciton
    'optimizer': optimizer, # torch optimizer
    'tensorboardParams': {
        'logdir': "../../runs/DQN_CarRacing_v3",
        'tag': "Averaged Returns/CNN_V2_lr=1e-3"
    },
}

# Initialize Actor-Critic Mehtod
DeepQN = DQN(**params_dict)

# load pretrained model
DeepQN.load("../../saved_models/CarRacing_v3/DQN_lr1e-3.obj")

# TRAIN Agent
DeepQN.train(10000)

DeepQN.save("../../saved_models/CarRacing_v3/DQN_lr1e-3.obj")
