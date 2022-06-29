
import sys
sys.path.append("../") # to import module

# PyTorch
import torch
import torch.optim as optim

# import model
from ActorCritic.models import ANN_V1
from ActorCritic.ActorCritic import ActorCritic

# Environment 
from envs.maze.Maze_Solver import MazeSolverEnv

MAX_EPISODES = 10000
MAX_TIMESTEPS = 1000

ALPHA = 0.1e-3 # learning rate
GAMMA = 0.99 # discount rate
epsilon = 0 # for epsilon greedy action

# device to use
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# set environment
env = MazeSolverEnv()

# set ActorCritic
num_actions = env.num_action
num_states = env.num_obs

ACmodel = ANN_V1(num_states, num_actions).to(device)
optimizer = optim.Adam(ACmodel.parameters(), lr=ALPHA)

ActorCritic_parameters = {
    'device': device, # device to use, 'cuda' or 'cpu'
    'env': env, # environment like gym
    'model': ACmodel, # torch models for policy and value funciton
    'optimizer': optimizer, # torch optimizer
    'maxTimesteps': MAX_TIMESTEPS, # maximum timesteps agent take 
    'discount_rate': GAMMA, # step-size for updating Q value
    'epsilon': epsilon 
}

# Initialize Actor-Critic Mehtod
AC = ActorCritic(**ActorCritic_parameters)

# TRAIN Agent
AC.train(MAX_EPISODES, useTensorboard=True, tensorboardTag="maze_ANN_V1")

