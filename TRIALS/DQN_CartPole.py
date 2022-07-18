
import sys
sys.path.append("../") # to import module

# Environment 
import gym

# set environment
env = gym.make(gym_name)

env.reset()
for i in range(100):
    env.step(0) 
    env.render()

env.close()
