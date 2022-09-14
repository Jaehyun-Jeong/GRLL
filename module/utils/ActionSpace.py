import numpy as np
import math
from random import choice


# OpenAI gym Box like action space class
class ActionSpace():

    def __init__(
            self,
            high: np.ndarray,
            low: np.ndarray):

        self.dtype = high.dtype  # Data type of each element
        self.high = high  # Biggest values of each element
        self.low = low  # Smallest values of each element
        self.shape = high.shape  # Shape of Actions

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
        return ~np.isin(
                self.low,
                np.array([math.inf, -math.inf]))

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
