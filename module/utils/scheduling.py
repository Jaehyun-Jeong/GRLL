import math


class Scheduling():

    def __init__(
            self,
            start: float,
            end: float,
            decay: int,):

        self.start = start
        self.end = end
        self.decay = decay


# Exponential Scheduling
class expScheduling(Scheduling):

    def __init__(
            self,
            start: float = 0.99,
            end: float = 0.0001,
            decay: int = 100000,):

        super().__init__(
                start=start,
                end=end,
                decay=decay,)

    def __call__(
            self,
            steps_done: int) -> float:

        eps_threshold = \
            self.end + (self.start - self.end) * \
            math.exp(-1. * steps_done / self.decay)

        return eps_threshold


# Linear Scheduling
class linScheduling(Scheduling):

    def __init__(
            self,
            start: float = 0.99,
            end: float = 0.0001,
            decay: int = 100000,):

        super().__init__(
                start=start,
                end=end,
                decay=decay,)

    def __call__(
            self,
            steps_done: int) -> float:

        eps_threshold = \
            self.start - steps_done * (self.start - self.end) / self.decay

        return eps_threshold
