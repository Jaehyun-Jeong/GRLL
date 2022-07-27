import sys
sys.path.append("../") # to import module

# PyTorch
import torch
import torch.optim as optim

# import model
from module.ValueBased.models import ANN_V2
from module.ValueBased import DQN

# Environment 
from module.envs.CarRacing import RacingEnv_v0

MAX_EPISODES = 20
MAX_TIMESTEPS = 1000
MAX_REPLAYMEMORY = 10000

ALPHA = 0.0001 # learning rate
GAMMA = 0.99 # discount rate

# device to use
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# set environment
env = RacingEnv_v0()

# set ActorCritic
num_actions = env.num_actions
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
    'useTensorboard': True,
    'tensorboardParams': {
        'logdir': "./runs/DQN_CarRacing_v0",
        'tag': "Averaged Returns (from 10 tests)"     
    },
    'eps': { # for epsilon scheduling
        'start': 0.99,
        'end': 0.00001,
        'decay': 1000
    },
    'policy': {
        'train': 'eps-stochastic',
        'test': 'stochastic' 
    }
}

# Initialize Actor-Critic Mehtod
DeepQN = DQN(**params_dict)

# TRAIN Agent
DeepQN.train(MAX_EPISODES)

# save model
DeepQN.save("./saved_models/DQN_RacingEnv_v0.obj")
