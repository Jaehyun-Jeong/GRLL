import sys
sys.path.append("../") # to import module
from itertools import product

# PyTorch
import torch
import torch.optim as optim

# import model
from module.ActorCritic.models import ANN_V2
from module.ActorCritic import onestep_ActorCritic

# Environment 
import gym

# device to use
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# set environment
gym_name = 'MountainCar-v0'
env = gym.make(gym_name)

# set ActorCritic
num_actions = env.action_space.n
num_states = env.observation_space.shape[0]
ActorCritic_model = ANN_V2(num_states, num_actions).to(device)
optimizer = optim.Adam(ActorCritic_model.parameters(), lr=0.1e-3)

# Initialize Actor-Critic Mehtod
onestep_AC = onestep_ActorCritic(
    device=device,
    env=env,
    model=ActorCritic_model,
    optimizer=optimizer
)

# TRAIN Agent
onestep_AC.train(maxEpisodes=3000, testPer=1, useTensorboard=True, tensorboardTag="ActorCritic_AL"+str(0.1e-3)+"_GA"+str(0.99)+"_"+gym_name)
