from gym.spaces import Box, Discrete
import numpy as np


def fromDiscrete(
        space: Discrete  # Discrete Action Space from gym
        ):

    # Biggiest index of action is space.n - 1
    # Because space.n is a size of action space
    high = np.array([space.n-1], dtype=space.dtype)
    low = np.array([0], dtype=space.dtype)

    return high, low


def fromBox(
        space: Box  # Box Action Space from gym
        ):

    high = space.high
    low = space.low

    return high, low
