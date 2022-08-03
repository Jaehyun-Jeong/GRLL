import sys
sys.path.append("../") # to import module

# PyTorch
import torch
import torch.optim as optim

# import model
from module.ValueBased.models import CNN_V2
from module.ValueBased import DQN

# Environment
from module.envs.CarRacing import RacingEnv_v3

MAX_EPISODES = 10000
MAX_TIMESTEPS = 100000
MAX_REPLAYMEMORY = 10000

ALPHA = 1e-3 # learning rate
GAMMA = 0.99 # discount rate

# device to use
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# set environment
env = RacingEnv_v3()

# set ActorCritic
num_actions = env.num_actions
num_states = env.num_obs

model = CNN_V2(num_states, num_actions).to(device)
optimizer = optim.Adam(model.parameters(), lr=ALPHA)

params_dict = {
    'device': device, # device to use, 'cuda' or 'cpu'
    'trainEnv': env, # environment like gym
    'testEnv': env, # environment like gym
    'model': model, # torch models for policy and value funciton
    'optimizer': optimizer, # torch optimizer
    'maxTimesteps': MAX_TIMESTEPS, # maximum timesteps agent take 
    'discount_rate': GAMMA, # step-size for updating Q value
    'maxMemory': MAX_REPLAYMEMORY,
    'numBatch': 100,
    'useTensorboard': True,
    'tensorboardParams': {
        'logdir': "./runs/DQN_CarRacing_v0",
        'tag': "Averaged Returns/CNN lr=1e-3"
    },
    'policy': {
        'train': 'stochastic',
        'test': 'stochastic' 
    },
    'isRender': {
        'train': False,
        'test': True,
    }
}

# Initialize Actor-Critic Mehtod
DeepQN = DQN(**params_dict)

# load pretrained model
# DeepQN.load("./saved_models/CarRacing_v2/DQN_lr1e-3.obj")

# TRAIN Agent
DeepQN.train(MAX_EPISODES)

# save model
DeepQN.save("./saved_models/CarRacing_v2/DQN_lr1e-3.obj")
