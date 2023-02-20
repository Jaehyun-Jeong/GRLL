from typing import Tuple
import numpy as np

import torch

from grll.utils.ActionSpace.ActionSpace import ActionSpace


class CalculatorEnv_v0():

    def __init__(self):

        self.num_actions = 110  # From -9 to 100
        # OBS LIST
        # 1. first_number
        # 2. second_number
        # 3. add
        # 4. sub
        # 5. mul
        # 6. answer
        self.num_obs = 6

    def step(self, action: torch.Tensor) \
            -> Tuple[np.ndarray, float, bool, torch.Tensor]:
        pass

    def reset(self) -> Tuple[np.ndarray, list]:
        pass

    def close(self):
        pass

    def render(self):
        raise NotImplementedError("This Environment isn't supporting rendering option!")
