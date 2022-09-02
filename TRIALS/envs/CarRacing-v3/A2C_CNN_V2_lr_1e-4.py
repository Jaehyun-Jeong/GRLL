import sys
sys.path.append("../../../")  # to import module

import torch
import torch.optim as optim
from module.PG.models import CNN_V2_shared
from module.PG import A2C
from module.envs.CarRacing import RacingEnv_v3

TRAIN_TIMESTEPS = 100000
MAX_TIMESTEPS = 100000

ALPHA = 1e-4  # learning rate
GAMMA = 0.99  # discount rate

# device to use
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# set environment
trainEnv = RacingEnv_v3(
        stackSize=10,
        ExploringStarts=True)

testEnv = RacingEnv_v3(stackSize=10)
num_actions = trainEnv.num_actions
num_states = testEnv.num_obs

model = CNN_V2_shared(num_states, num_actions).to(device)
optimizer = optim.Adam(model.parameters(), lr=ALPHA)

params_dict = {
    'device': device,  # device to use, 'cuda' or 'cpu'
    'trainEnv': trainEnv,  # environment like gym
    'testEnv': testEnv,  # environment like gym
    'model': model,  # torch models for policy and value funciton
    'optimizer': optimizer,  # torch optimizer
    'maxTimesteps': MAX_TIMESTEPS,  # maximum timesteps agent take
    'discount_rate': GAMMA,  # step-size for updating Q value
    'useTensorboard': True,
    'tensorboardParams': {
        'logdir': "../../runs/A2C_CarRacing_v3",
        'tag': "Averaged Returns/CNN_V2_lr=1e-4"
    },
    'eps': {  # for epsilon scheduling
        'start': 0.99,
        'end': 0.00001,
        'decay': 300000
    },
    'policy': {
        'train': 'stochastic',
        'test': 'greedy',
    },
}

# Initialize Actor-Critic Mehtod
Trainer = A2C(**params_dict)

# TRAIN Agent
Trainer.train(TRAIN_TIMESTEPS)

# save model
Trainer.save("../../saved_models/CarRacing_v3/A2C_lr1e-4.obj")
