import sys
sys.path.append("../../../") # to import module

# PyTorch
import torch
import torch.optim as optim

# import model
from module.PolicyGradient.models import ANN_V1
from module.PolicyGradient import onestep_ActorCritic

# Environment 
import gym

# device to use
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# set environment
gym_name = 'CartPole-v0'
env = gym.make(gym_name)

# set ActorCritic
num_actions = env.action_space.n
num_states = env.observation_space.shape[0]

ActorCritic_model = ANN_V1(num_states, num_actions)
optimizer = optim.Adam(ActorCritic_model.parameters(), lr=0.1e-3)

# Initialize Actor-Critic Mehtod
onestep_AC = onestep_ActorCritic(
    device=device,
    env=env,
    model=ActorCritic_model,
    optimizer=optimizer,
    isRender={
        'train': False,
        'test': False,
    },
)

# TRAIN Agent
onestep_AC.train(maxEpisodes=100000, testPer=100)
