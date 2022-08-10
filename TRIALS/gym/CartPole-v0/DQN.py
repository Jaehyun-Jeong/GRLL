
import sys
sys.path.append("../../../") # to import module

# 시간 측정
from datetime import datetime
startTime = datetime.now()

# PyTorch
import torch
import torch.optim as optim

# import model
from module.ValueBased.models import ANN_V3
from module.ValueBased import DQN

# Environment 
import gym
env = gym.make('CartPole-v0')
num_actions = env.action_space.n
num_states = env.observation_space.shape[0]

# set ActorCritic
model = ANN_V3(num_states, num_actions)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Initialize Actor-Critic Mehtod
DeepQN = DQN(
    model=model, # torch models for policy and value funciton
    env=env,
    optimizer=optimizer, # torch optimizer
    maxMemory=100000,
    numBatch=32,
    verbose=0,
)

print(f"Init Time: {datetime.now() - startTime}")

startTrainTime = datetime.now()

# TRAIN Agent
DeepQN.train(trainTimesteps=100000, testSize=0)

print(f"Train Time: {datetime.now() - startTrainTime}")
print(DeepQN.test(testSize=10))
