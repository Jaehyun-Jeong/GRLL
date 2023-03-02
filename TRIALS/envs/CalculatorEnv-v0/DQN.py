import sys
sys.path.append("../../../") # to import module

# PyTorch
import torch
import torch.optim as optim

import math

# import model
from grll.VB.models import ANN_Cal
from grll.VB import DQN

# Environment
from grll.envs.Calculator import CalculatorEnv_v0

TRAIN_TIMESTEPS = int(1e8)
MAX_TIMESTEPS = 1000
MAX_REPLAYMEMORY = 50000

ALPHA = 0.0001 # learning rate
GAMMA = 0 # discount rate

# device to use
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# set environment
env = CalculatorEnv_v0()
num_actions = env.num_actions
num_states = env.num_obs

model = ANN_Cal(num_states, num_actions).to(device)
optimizer = optim.Adam(model.parameters(), lr=ALPHA)

params_dict = {
    'device': device, # device to use, 'cuda' or 'cpu'
    'env': env,
    'model': model, # torch models for policy and value funciton
    'optimizer': optimizer, # torch optimizer
    'maxTimesteps': MAX_TIMESTEPS, # maximum timesteps agent take 
    'discount': GAMMA, # step-size for updating Q value
    'maxMemory': MAX_REPLAYMEMORY,
    'numBatch': 32,
    'useTensorboard': True,
    'tensorboardParams': {
        'logdir': "../../runs/DQN_Calculator_v0",
        'tag': "Averaged Returns/ANN_Cal_lr=1e-4"
    },
    'actionParams': {
        # for DISCRETE
        'algorithm': "greedy",  # greedy, stochastic
        'exploring': "epsilon",  # epsilon, None
        'exploringParams': {
            'start': 0.95,
            'end': 0.05,
            'decay': 1000000,
        },
    },
    'verbose': 2,
}

# Initialize Actor-Critic Mehtod
DeepQN = DQN(**params_dict)
DeepQN.train(TRAIN_TIMESTEPS, 1000, 100)
