import sys
sys.path.append("../") # to import module

# PyTorch
import torch
import torch.optim as optim

# import model
from ActorCritic.models import ANN_V2
from ActorCritic.ActorCritic import ActorCritic

# Environment 
import gym

MAX_EPISODES = 10000
MAX_TIMESTEPS = 1000

ALPHA = 3e-4 # learning rate
GAMMA = 0.99 # step-size

# device to use
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# set environment
env = gym.make('CartPole-v0')
#env = gym.make('MountainCar-v0')

# set ActorCritic
num_actions = env.action_space.n
num_states = env.observation_space.shape[0]
ACmodel = ANN_V2(num_states, num_actions).to(device)
optimizer = optim.Adam(ACmodel.parameters(), lr=ALPHA)

ActorCritic_parameters = {
    'device': device, # device to use, 'cuda' or 'cpu'
    'env': env, # environment like gym
    'model': ACmodel, # torch models for policy and value funciton
    'optimizer': optimizer, # torch optimizer
    'maxTimesteps': MAX_TIMESTEPS, # maximum timesteps agent take 
    'stepsize': GAMMA # step-size for updating Q value
}

# Initialize Actor-Critic Mehtod
AC = ActorCritic(**ActorCritic_parameters)

# TRAIN Agent
AC.train(MAX_EPISODES)

