import math
import numpy as np

##################
# DISCRETE NOISE #
##################


# for epsilon greedy
class epsilon():

    def __init__(
            self,
            start=0.99,
            end=0.0001,
            decay=10000):

        self._start = start
        self._end = end
        self._decay = decay

    def __call__(
            self,
            stepsDone: int
            ) -> float:

        threshold = \
            self._end + (self._start - self._end) * \
            math.exp(-1. * stepsDone / self._decay)

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
