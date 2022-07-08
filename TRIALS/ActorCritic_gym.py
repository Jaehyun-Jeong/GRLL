import sys
sys.path.append("../") # to import module
from itertools import product

# PyTorch
import torch
import torch.optim as optim

# import model
from ActorCritic.models import ANN_V2
from ActorCritic.ActorCritic import ActorCritic

# Environment 
import gym

MAX_EPISODES = 500
MAX_TIMESTEPS = 1000

ALPHA = 0.0001 # learning rate
GAMMA = 0.99 # discount rate
epsilon = 0

gym_name = 'CartPole-v0'

# device to use
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# set environment
env = gym.make(gym_name)

# set ActorCritic
num_actions = env.action_space.n
num_states = env.observation_space.shape[0]
ActorCritic_model = ANN_V2(num_states, num_actions).to(device)
optimizer = optim.Adam(ActorCritic_model.parameters(), lr=ALPHA)

ActorCritic_parameters = {
    'device': device, # device to use, 'cuda' or 'cpu'
    'env': env, # environment like gym
    'model': ActorCritic_model, # torch models for policy and value funciton
    'optimizer': optimizer, # torch optimizer
    'maxTimesteps': MAX_TIMESTEPS, # maximum timesteps agent take 
    'discount_rate': GAMMA, # step-size for updating Q value
}

# Initialize Actor-Critic Mehtod
RF = ActorCritic(**ActorCritic_parameters)

# TRAIN Agent
RF.train(MAX_EPISODES, testPer=1, useTensorboard=True, tensorboardTag="ActorCritic_AL"+str(ALPHA)+"_GA"+str(GAMMA)+"_"+gym_name)
