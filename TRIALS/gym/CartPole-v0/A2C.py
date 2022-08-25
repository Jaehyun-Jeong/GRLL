import sys
sys.path.append("../../../") # to import module

# PyTorch
import torch
import torch.optim as optim

# import model
from module.PolicyGradient.models import ANN_V3
from module.PolicyGradient import A2C

# Environment
import gym
env = gym.make('CartPole-v0')
num_actions = env.action_space.n
num_states = env.observation_space.shape[0]

A2C_model = ANN_V3(num_states, num_actions)
optimizer = optim.Adam(ActorCritic_model.parameters(), lr=1e-4)

# Initialize Actor-Critic Mehtod
onestep_AC = A2C(
    env=env,
    model=A2C_model,
    optimizer=optimizer,
)

# TRAIN Agent
onestep_AC.train(
        trainTimesteps=1000000,
        testSize=0)

# save model
onestep_AC.save("../../saved_models/CartPole-v0/onestep_AC.obj")
