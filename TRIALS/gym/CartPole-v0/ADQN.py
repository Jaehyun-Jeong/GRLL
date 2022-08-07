import sys
sys.path.append("../../../") # to import module

# PyTorch
import torch
import torch.optim as optim

# import model
from module.ValueBased.models import ANN_V2
from module.ValueBased import ADQN

# Environment 
import gym

MAX_EPISODES = 3000
MAX_TIMESTEPS = 1000
MAX_REPLAYMEMORY = 10000

ALPHA = 0.0001 # learning rate
GAMMA = 0.99 # discount rate

gym_name = 'CartPole-v1'

# device to use
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# set environment
env = gym.make(gym_name)

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
    'maxTimesteps': MAX_TIMESTEPS, # maximum timesteps agent take 
    'discount_rate': GAMMA, # step-size for updating Q value
    'maxMemory': MAX_REPLAYMEMORY,
    'numBatch': 64,
    'eps': { # for epsilon scheduling
        'start': 0.99,
        'end': 0.00001,
        'decay': 100000
    },
    'useTensorboard': True,
    'tensorboardParams': {
        'logdir': '../../runs/ADQN',
        'tag': 'CartPole'
    }
}

# Initialize Actor-Critic Mehtod
averagedDQN = ADQN(**params_dict)

# TRAIN Agent
averagedDQN.train(MAX_EPISODES)
