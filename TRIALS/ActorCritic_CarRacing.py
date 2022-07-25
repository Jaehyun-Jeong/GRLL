import sys
sys.path.append("../") # to import module

# PyTorch
import torch
import torch.optim as optim

# import model
from module.ActorCritic.models import ANN_V1
from module.ActorCritic import onestep_ActorCritic

# Environment 
from module.envs.CarRacing import RacingEnv_v0

# device to use
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# set environment
gym_name = 'RacingEnv-v0'
env = RacingEnv_v0()

# set ActorCritic
num_actions = env.num_actions
num_states = env.num_obs

ActorCritic_model = ANN_V1(num_states, num_actions).to(device)
optimizer = optim.Adam(ActorCritic_model.parameters(), lr=0.1e-3)

# Initialize Actor-Critic Mehtod
onestep_AC = onestep_ActorCritic(
    device=device,
    env=env,
    model=ActorCritic_model,
    optimizer=optimizer,
    isRender={
        'train': False,
        'test': True, 
    },
    useTensorboard=True,
    tensorboardParams={
        'logdir': "./runs/onestep_ActorCritic_CarRacing",
        'tag': "Averaged Returns (from 10 tests)"     
    }
)

# TRAIN Agent
onestep_AC.train(maxEpisodes=1000)

onestep_AC.save("./saved_models/onestep_ActorCritic_RacingEnv_v0.obj")
