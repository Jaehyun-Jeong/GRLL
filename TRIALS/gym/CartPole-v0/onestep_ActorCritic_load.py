import sys
sys.path.append("../../../") # to import module

# PyTorch
import torch
import torch.optim as optim

# import model
from module.PolicyGradient.models import ANN_V2
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

ActorCritic_model = ANN_V2(num_states, num_actions)
optimizer = optim.Adam(ActorCritic_model.parameters(), lr=1e-4)

# Initialize Actor-Critic Mehtod
onestep_AC = onestep_ActorCritic(
    device=device,
    env=env,
    model=ActorCritic_model,
    optimizer=optimizer,
    useTensorboard=True,
    tensorboardParams={
        'logdir': "../../runs/onestep_AC_CartPole-v0",
        'tag': "Averaged Returns/ANN_V2_lr=1e-4"
    },
)

# load model
onestep_AC.load("../../saved_models/CartPole-v0/onestep_AC.obj")

# TRAIN Agent
onestep_AC.train(1000000)

# save model
onestep_AC.save("../../saved_models/CartPole-v0/onestep_AC.obj")
