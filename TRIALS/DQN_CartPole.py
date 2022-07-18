
import mujoco_py
import gym

import os
os.environ.get("LD_LIBRARY_PATH", "")

env = gym.make("HalfCheetah-v2")
env.reset()

for i in range(100):
    env.step(env.action_space.sample())
    env.render()
