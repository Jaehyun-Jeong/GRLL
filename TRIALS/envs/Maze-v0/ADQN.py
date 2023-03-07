
import sys
sys.path.append("../../../") # to import module

# PyTorch
import torch
import torch.optim as optim

# import model
from grll.VB.models import ANN_V2
from grll.VB.ADQN import ADQN

# Environment 
from grll.envs.Maze import MazeEnv_v0

MAX_EPISODES = 3000
MAX_TIMESTEPS = 1000
MAX_REPLAYMEMORY = 100000

ALPHA = 0.0001 # learning rate
GAMMA = 0.99 # discount rate

gym_name = 'maze'

# device to use
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# set environment
env = MazeEnv_v0()

# set ActorCritic
num_actions = env.num_action
num_states = env.num_obs
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
    'policy': {
        'train': 'eps-stochastic',
        'test': 'stochastic',
    },
    'isRender': {
        'train': False,
        'test': False
    }
}

# Initialize Actor-Critic Mehtod
averagedDQN = ADQN(**params_dict)

# TRAIN Agent
averagedDQN.train(MAX_EPISODES, testPer=1)
