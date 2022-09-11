import sys
sys.path.append("../../../") # to import module

# PyTorch
import torch
import torch.optim as optim

# import model
from module.VB.models import ANN_V4
from module.VB import DQN

# Environment
from module.envs.CarRacing import RacingEnv_v3

TRAIN_TIMESTEPS = int(1e7)
MAX_TIMESTEPS = 100000
MAX_REPLAYMEMORY = 10000

ALPHA = 1e-4 # learning rate
GAMMA = 0.99 # discount rate

# device to use
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# set environment
trainEnv = RacingEnv_v3(
        stackSize=1,
        imgSize=(32, 32),
        isFlatten=True,
        ExploringStarts=True)

testEnv = RacingEnv_v3(
        stackSize=1,
        imgSize=(32, 32),
        isFlatten=True,
        ExploringStarts=False)

num_actions = trainEnv.num_actions
num_states = testEnv.num_obs

model = ANN_V4(num_states, num_actions).to(device)
optimizer = optim.Adam(model.parameters(), lr=ALPHA)

params_dict = {
    'device': device, # device to use, 'cuda' or 'cpu'
    'trainEnv': trainEnv, # environment like gym
    'testEnv': testEnv, # environment like gym
    'model': model, # torch models for policy and value funciton
    'optimizer': optimizer, # torch optimizer
    'maxTimesteps': MAX_TIMESTEPS, # maximum timesteps agent take 
    'discount': GAMMA, # step-size for updating Q value
    'maxMemory': MAX_REPLAYMEMORY,
    'numBatch': 128,
    'useTensorboard': True,
    'tensorboardParams': {
        'logdir': "../../runs/DQN_CarRacing_v3",
        'tag': "Averaged Returns/ANN_V4_lr=1e-4"
    },
    'eps': { # for epsilon scheduling
        'start': 0.99,
        'end': 0.00001,
        'decay': 300000
    },
    'trainStarts': 10
}

# Initialize Actor-Critic Mehtod
DeepQN = DQN(**params_dict)

# load pretrained model
# DeepQN.load("./saved_models/CarRacing_v2/DQN_lr1e-3.obj")

# TRAIN Agent
DeepQN.train(TRAIN_TIMESTEPS)

# save model
DeepQN.save("../../saved_models/CarRacing_v3/ANN_V4_lr1e-4.obj")
