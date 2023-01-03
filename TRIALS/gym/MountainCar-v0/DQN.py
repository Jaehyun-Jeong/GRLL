import sys
sys.path.append("../../../") # to import module

# 시간 측정
from datetime import datetime
startTime = datetime.now()

# PyTorch
import torch
import torch.optim as optim

# import model
from grll.VB.models import ANN_V2
from grll.VB import DQN

# Environment 
import gym

MAX_TIMESTEPS = 1000
MAX_REPLAYMEMORY = 10000

ALPHA = 0.0001 # learning rate
GAMMA = 0.99 # discount rate
epsilon = 0.3

gym_name = 'MountainCar-v0'

# device to use
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

env = gym.make('MountainCar-v0')
num_actions = env.action_space.n
num_states = env.observation_space.shape[0]

# set ActorCritic
model = ANN_V2(num_states, num_actions)
optimizer = optim.Adam(model.parameters(), lr=ALPHA)

# Initialize Actor-Critic Mehtod
DeepQN = DQN(
    model=model, # torch models for policy and value funciton
    env=env,
    optimizer=optimizer, # torch optimizer
    maxTimesteps=MAX_TIMESTEPS,
    maxMemory=MAX_REPLAYMEMORY,
    numBatch=32,
    verbose=1,
)

print(f"Init Time: {datetime.now() - startTime}")

startTrainTime = datetime.now()

# TRAIN Agent
DeepQN.train(trainTimesteps=1000000, testPer=10000)
