import sys
sys.path.append("../../../") # to import module

# PyTorch
import torch
import torch.optim as optim

# import model
from grll.PG.models import ANN_V2
from grll.PG import REINFORCE

# Environment 
import gym

TRAIN_TIMESTEPS = 1000000
MAX_TIMESTEPS = 1000

ALPHA = 3e-4 # learning rate
GAMMA = 0.99 # discount rate
epsilon = 0.7 

gym_name = 'CartPole-v0'

# device to use
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

env = gym.make(gym_name)

# set ActorCritic
num_actions = env.action_space.n
num_states = env.observation_space.shape[0]
REINFORCE_model = ANN_V2(num_states, num_actions).to(device)
optimizer = optim.Adam(REINFORCE_model.parameters(), lr=ALPHA)

REINFORCE_parameters= {
    'device': device, # device to use, 'cuda' or 'cpu'
    'env': env, # environment like gym
    'model': REINFORCE_model, # torch models for policy and value funciton
    'optimizer': optimizer, # torch optimizer
}

# Initialize Actor-Critic Mehtod
RF = REINFORCE(**REINFORCE_parameters)

# TRAIN Agent
RF.train(TRAIN_TIMESTEPS, testPer=10000)

