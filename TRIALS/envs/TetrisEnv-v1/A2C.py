import sys
sys.path.append("../../../")  # to import module

import torch
import torch.optim as optim
from grll.PG.models import ANN_V4
from grll.PG import A2C
from grll.envs.Tetris import TetrisEnv_v1

TRAIN_TIMESTEPS = int(1e8)
MAX_TIMESTEPS = 100000

ALPHA = 1e-4  # learning rate
GAMMA = 0.99  # discount rate

# device to use
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# set environment
env = TetrisEnv_v1()

num_actions = env.num_actions
num_states = env.num_obs

model = ANN_V4(num_states, num_actions).to(device)
optimizer = optim.Adam(model.parameters(), lr=ALPHA)

params_dict = {
    'device': device,  # device to use, 'cuda' or 'cpu'
    'env': env,
    'model': model,  # torch models for policy and value funciton
    'optimizer': optimizer,  # torch optimizer
    'maxTimesteps': MAX_TIMESTEPS,  # maximum timesteps agent take
    'discount': GAMMA,  # step-size for updating Q value
    'useTensorboard': True,
    'tensorboardParams': {
        'logdir': "../../runs/A2C_TetrisEnv_v1",
        'tag': "Averaged Returns/ANN_V4_lr=1e-4"
    },
    'actionParams': {
        'algorithm': 'stochastic',
        'exploring': 'epsilon',
        'exploringParams': {
            'start': 0.99,
            'end': 0.00001,
            'decay': 300000
        }
    },
}

load_params_dict = {
    'device': device,  # device to use, 'cuda' or 'cpu'
    'env': env,
    'model': model,  # torch models for policy and value funciton
    'optimizer': optimizer,  # torch optimizer
    'useTensorboard': True,
    'tensorboardParams': {
        'logdir': "../../runs/A2C_TetrisEnv_v1",
        'tag': "Averaged Returns/ANN_V4_lr=1e-4"
    },
}

# Initialize Actor-Critic Mehtod
Trainer = A2C(**params_dict)

separated_train_step = int(TRAIN_TIMESTEPS / 1000)
for i in range(1000):

    # TRAIN Agent
    Trainer.train(separated_train_step)

    # save model
    Trainer.save(
            "../../saved_models/TetrisEnv_v1/" +
            f"A2C_ANN_V4_lr1e-4_step_{separated_train_step * (i+1)}.obj")

    Trainer = A2C(**load_params_dict)

    Trainer.load(
            "../../saved_models/TetrisEnv_v1/" +
            f"A2C_ANN_V4_lr1e-4_step_{separated_train_step * (i+1)}.obj")
