import math
import numpy as np

from module.utils.scheduling import expScheduling, linScheduling

##################
# DISCRETE NOISE #
##################

# for epsilon greedy
class epsilon():

    def __init__(
            self,
            schedule=expSheduling(  # exponential, linear
                    start=0.99,
                    end=0.0001,
                    decay=10000
                    ),
            ):

        self.schedule = schedule

    def __call__(
            self,
            stepsDone: int
            ) -> float:

        threshold = self.schedule(stepsDone)

        return threshold


####################
# CONTINUOUS NOISE #
####################


class NormalNoise():

    def __init__(
            self,
            mean: float,
            sigma: float):

        self._mu = mean
        self._sigma = sigma

    def __call__(self) -> np.ndarray:
        return np.random.normal(self._mu, self._sigma)
