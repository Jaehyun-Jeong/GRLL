from typing import Union, Dict

import random
import numpy as np

# PyTorch
import torch


class ActionValue():

    def __init__(
            self,
            policyModel: torch.nn.Module,
            actionType: str,  # Discrete, Continuous
            actionParams: Dict[str, Union[int, float]]):

        # Set policy
        if actionType == 'Discrete':
            self.policy = DiscretePolicy(**actionParams)
        elif actionType == 'Continuous':
            self.policy = ContinuousPolicy(**actionParams)
        else:
            raise ValueError(
                "actionType only for Discrete and Continuous Action")
