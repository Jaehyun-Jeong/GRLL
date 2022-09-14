from typing import Union, Dict

import numpy as np

# PyTorch
import torch

# module
from module.Policy import DiscretePolicy, ContinuousPolicy
from module.utils.ActionSpace import ActionSpace


class ActionValue():

    def __init__(
            self,
            policyModel: torch.nn.Module,
            actionSpace: ActionSpace,
            actionParams: Dict[str, Union[int, float]]):

        self.actionSpace = actionSpace

        # Set policy
        if self.actionSpace.actionType == 'Discrete':
            self.policy = DiscretePolicy(**actionParams)
        elif self.actionSpace.actionType == 'Continuous':
            self.policy = ContinuousPolicy(**actionParams)
        else:
            raise ValueError(
                "actionType only for Discrete and Continuous Action")

    # Get Action Value from state
    def get_value(
            self,
            s: Union[torch.Tensor, np.ndarray],
            ) -> torch.Tensor:

        s = torch.Tensor(s).to(self.device).unsqueeze(0)
        _, ActionValue = self.model.forward(s)
        ActionValue = ActionValue.squeeze(0)

        return ActionValue

    # Get Action from State s
    def get_action(
            self,
            s: Union[torch.Tensor, np.ndarray],
            ) -> torch.Tensor:

        ActionValue = self.get_value(s)

        return self.policy.get_action(
                ActionValue
                )
