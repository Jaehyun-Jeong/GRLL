import sys
sys.path.append("../../../") # to import module

# PyTorch
import torch
import torch.optim as optim

# import model
from module.ValueBased.models import ANN_V2
from module.ValueBased import DQN

# Environment 
import gym

MAX_EPISODES = 500
MAX_TIMESTEPS = 1000
MAX_REPLAYMEMORY = 10000

ALPHA = 0.0001 # learning rate
GAMMA = 0.99 # discount rate

# device to use
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# set environment
env = gym.make('CartPole-v0')

# set ActorCritic
num_actions = env.action_space.n
num_states = env.observation_space.shape[0]
model = ANN_V2(num_states, num_actions).to(device)
optimizer = optim.Adam(model.parameters(), lr=ALPHA)

params_dict = {
    'device': device, # device to use, 'cuda' or 'cpu'
    'env': env, # environment like gym
    'model': model, # torch models for policy and value funciton
    'optimizer': optimizer, # torch optimizer
    'useTensorboard': True,
    'tensorboardParams': {
        'logdir': "../../runs/DQN_CartPole-v0",
        'tag': "Averaged Returns (from 10 tests)"     
    },
}

# Initialize Actor-Critic Mehtod
DeepQN = DQN(**params_dict)

# load pretrained model
DeepQN.load("../../saved_models/CartPole-v0/DQN_lr1e-3.obj")

# TRAIN Agent
DeepQN.train(MAX_EPISODES)

# save model
DeepQN.save("../../saved_models/CartPole-v0/DQN_lr1e-3.obj")
