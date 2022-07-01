import sys
sys.path.append("../") # to import module
from itertools import product

# PyTorch
import torch
import torch.optim as optim

# import model
from ActorCritic.models import ANN_V2
from ActorCritic.ActorCritic import ActorCritic

# Environment 
import gym

MAX_EPISODES = 3000
MAX_TIMESTEPS = 1000

ALPHAs = [0.01, 0.001, 0.0001] # learning rate
GAMMAs = [0.9, 0.95, 0.99] # discount rate
epsilon = 0

#gym_list = ['Acrobot-v1', 'MountainCar-v0', 'CartPole-v0']
gym_list = ['MountainCar-v0']

# device to use
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

for gym_name in gym_list:
    for ALPHA, GAMMA in product(ALPHAs, GAMMAs):

        # set environment
        env = gym.make(gym_name)

        # set ActorCritic
        num_actions = env.action_space.n
        num_states = env.observation_space.shape[0]
        ActorCritic_model = ANN_V2(num_states, num_actions).to(device)
        optimizer = optim.Adam(ActorCritic_model.parameters(), lr=ALPHA)

        ActorCritic_parameters = {
            'device': device, # device to use, 'cuda' or 'cpu'
            'env': env, # environment like gym
            'model': ActorCritic_model, # torch models for policy and value funciton
            'optimizer': optimizer, # torch optimizer
            'maxTimesteps': MAX_TIMESTEPS, # maximum timesteps agent take 
            'discount_rate': GAMMA, # step-size for updating Q value
            'epsilon': epsilon,
        }

        # Initialize Actor-Critic Mehtod
        RF = ActorCritic(**ActorCritic_parameters)

        # TRAIN Agent
        RF.train(MAX_EPISODES, testPer=1, useTensorboard=True, tensorboardTag="ActorCritic_AL"+str(ALPHA)+"_GA"+str(GAMMA)+"_"+gym_name)
