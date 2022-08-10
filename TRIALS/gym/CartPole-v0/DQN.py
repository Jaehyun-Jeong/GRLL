
import sys
sys.path.append("../../../") # to import module

# PyTorch
import torch
import torch.optim as optim

# import model
from module.ValueBased.models import ANN_V3
from module.ValueBased import DQN

TRAIN_TIMESTEPS = 100000
MAX_TIMESTEPS = 100000
MAX_REPLAYMEMORY = 10000

ALPHA = 1e-3 # learning rate
GAMMA = 0.99 # discount rate

# device to use
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

# Environment 
import gym

# set environment
env = gym.make('CartPole-v0')

# set ActorCritic
num_actions = env.action_space.n
num_states = env.observation_space.shape[0]
model = ANN_V3(num_states, num_actions)
optimizer = optim.Adam(model.parameters(), lr=ALPHA)

params_dict = {
    'device': device, # device to use, 'cuda' or 'cpu'
    'model': model, # torch models for policy and value funciton
    'env': env,
    'optimizer': optimizer, # torch optimizer
    'maxTimesteps': MAX_TIMESTEPS, # maximum timesteps agent take 
    'discount_rate': GAMMA, # step-size for updating Q value
    'maxMemory': MAX_REPLAYMEMORY,
    'numBatch': 100,
    'isRender': {
        'train': True,
        'test': False,
    }
}

# Initialize Actor-Critic Mehtod
DeepQN = DQN(**params_dict)

# TRAIN Agent
DeepQN.train(TRAIN_TIMESTEPS)

DeepQN.test(testSize = 5)

'''
# save model
DeepQN.save("../../saved_models/CartPole-v0/DQN_lr1e-3.obj")
'''
