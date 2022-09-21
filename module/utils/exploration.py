from typing import Union

import numpy as np
from module.utils.scheduling import expScheduling, linScheduling

##################
# DISCRETE NOISE #
##################


# for epsilon greedy
class Epsilon():

    """
    parameters
        schedule: Schedule algorithm name to use
            e.g.) "exponential", "linear"
        start: Start epsilon value for epsilon greedy policy
        end: Final epsilon value for epsilon greedy policy
        decay: It determines how small epsilon is
    """

    def __init__(
            self,
            schedule="exponential",
            start=0.99,
            end=0.0001,
            decay=10000
            ):

        # Init Scheduling

        scheduleDict = {
                'exponential': expScheduling,
                'linear': linScheduling
                }

        self.schedule = scheduleDict[schedule](
                start=start,
                end=end,
                decay=decay,
                )

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

    """
    parameter
        mean: Mean of normal distribution
        sigma: Standard Deviation of normal distribution
    """

    def __init__(
            self,
            mean: float,
            sigma: float):

        self._mu = mean
        self._sigma = sigma

    def __call__(
            self,
            stepsDone: int,
            shape: Union[tuple, int],
            ) -> float:

        return np.random.normal(self._mu, self._sigma, shape)
