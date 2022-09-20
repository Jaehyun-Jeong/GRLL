import numpy as np
import math
from random import choice

# Check type of ActionSpace
# Gym
from gym.spaces import Discrete, Box

from module.utils.ActionSpace.gymConverter \
        import fromDiscrete, fromBox


# OpenAI gym Box like action space class
class ActionSpace():

    """
    1. parameters

    high: Biggest values of each element
    low: Smallest values of each element
    actionSpace:
        When it need converting from other module (like gym)
        use this paramter

        Warning:
            If you want to use this parameter,
            then high and low parameters have to be None (or not specified)

    2. USE CASE

    case1)
    use high, low parameters
    np.ndarray high and low automatically make ActionSpace

    case2)
    use actionSpace parameter

    Support spaces
        1. Box from gym
        2. Discrete from gym
    """

    def __init__(
            self,
            high: np.ndarray = None,
            low: np.ndarray = None,
            actionSpace=None):

        # If it has its own actionSpace
        if (high is None and low is None) and actionSpace is not None:

            if type(actionSpace) == Discrete:
                self.high, self.low = fromDiscrete(actionSpace)
            if type(actionSpace) == Box:
                self.high, self.low = fromBox(actionSpace)

            if not (type(actionSpace) in [Discrete, Box]):
                raise ValueError(
                    f"Supported Action Spaces are {str(Discrete)}, {str(Box)}")

        # When high and low given
        elif (high is not None and low is not None) and actionSpace is None:
            self.high = high  # Biggest values of each element
            self.low = low  # Smallest values of each element
        else:
            raise ValueError(
                    "")

        self.dtype = self.high.dtype  # Data type of each element
        self.shape = self.high.shape  # Shape of Actions

        # Check Validity
        # int or float
        if self.dtype in [np.int32, np.int64]:
            self.actionType = 'Discrete'
        elif self.dtype in [np.float32, np.float64]:
            self.actionType = 'Continuous'
        else:
            raise ValueError(
                    "Action Space data type should be float or integer")

        # Check data type and shape
        if self.high.dtype != self.low.dtype:
            raise ValueError(
                    "high and low have different data type!")
        if self.high.shape != self.low.shape:
            raise ValueError(
                    "high and low have different shape!")

    # Check the Validity of X
    def contains(
            self,
            X: np.ndarray,
            ) -> bool:

        if X.shape != self.shape:
            ValueError(f"{X} has different shape!")

        if False in (X <= self.high):
            return False

        if False in (X >= self.low):
            return False

        return True

    # Check each element have upper bound or not
    def bounded_above(self):
        return ~np.isin(
                self.high,
                math.inf)

    # Check each element have lower bound or not
    def bounded_below(self):
        return ~np.isin(
                self.low,
                math.inf)

    # Check each element have bound or not
    def is_bounded(self):

        above = self.bounded_above()
        below = self.bounded_below()

        return np.logical_or(above, below)

    # Sample Random Action
    def sample(self):

        if self.dtype in [np.float32, np.float64]:

            # Get action fron uniform distribution
            action = []
            for low, high in zip(self.low, self.high):
                action.append(np.random.uniform(
                    low=low,
                    high=high))

            return np.array(action)

        elif self.dtype in [np.int32, np.int64]:

            # Get action by random choice from integer range
            action = []
            for low, high in zip(self.low, self.high):
                action.append(choice(range(low, high)))

            return action

        else:
            ValueError("Action should be integer or float data type")
