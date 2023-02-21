from typing import Union, Dict

import numpy as np

# PyTorch
import torch

# module
from grll.Policy import DiscretePolicy, ContinuousPolicy
from grll.utils.ActionSpace import ActionSpace


class Value():

    """
    parameters
        model: torch.nn.Module based model for state_value, and action_value
        device: Device used for training, like Backpropagation
        optimizer: torch optimizer
        actionSapce: inner module ActionSpace class
        actionParameters={
            # for DISCRETE
            'algorithm': "greedy",  # greedy, stochastic
            'exploring': "epsilon",  # epsilon, None
            'exploringParams': {
                'start': 0.99,
                'end': 0.0001,
                'decay': 10000
            },

            # for CONTINUOUS
            'algorithm': "plain",  # plain
            'exploring': "normal",  # normal
            'exploringParams': {
                'mean': 0,
                'sigma': 1,
            }
        }
    """

    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        optimizer,
        actionSpace: ActionSpace,
        actionParams: Dict[str, Union[int, float, Dict]] = None,
        clippingParams: Dict[str, Union[int, float]] = {
            'pNormValue': 2,
            'maxNorm': 1,
        },
    ):

        # Initialize Parameter
        self.device = device

        # Load model and set to eval mod
        self.model = model.to(self.device)
        self.model.eval()

        self.optimizer = optimizer
        self.clippingParams = clippingParams
        self.actionSpace = actionSpace
        self.stepsDone = 0

        # Set policy
        if self.actionSpace.actionType == 'Discrete':

            # default actionParams
            if actionParams is None:
                actionParams = {
                    'algorithm': "greedy",  # greedy, stochastic
                    'exploring': "epsilon",  # epsilon, None
                    'exploringParams': {
                        'start': 0.99,
                        'end': 0.0001,
                        'decay': 10000
                    }
                }

            self.policy = DiscretePolicy(
                    **actionParams,
                    actionSpace=actionSpace)

        if self.actionSpace.actionType == 'Continuous':

            # default actionParams
            if actionParams is None:
                actionParams = {
                    'algorithm': "plain",  # greedy
                    'exploring': "normal",  # normal
                    'exploringParams': {
                        'mean': 0,
                        'sigma': 1,
                    }
                }

            self.policy = ContinuousPolicy(
                    **actionParams,
                    actionSpace=actionSpace)

        if self.actionSpace.actionType \
                not in ['Discrete', 'Continuous']:

            raise ValueError(
                "actionType only for Discrete and Continuous Action")

    # Update Weights
    def step(self, loss):

        # Calculate Gradient
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=self.clippingParams['maxNorm'],
                norm_type=self.clippingParams['pNormValue'],
                )

        # Backpropagation and count steps
        self.optimizer.step()
        self.stepsDone += 1

    # Returns a value of the state
    # (state value function in Reinforcement learning)
    def state_value(
            self,
            s: torch.Tensor,
            ) -> torch.Tensor:

        value, _ = self.model.forward(s)
        value = value.squeeze(dim=-1)

        return value

    # Get all Action Value as Tensor from state
    def action_value(
            self,
            s: Union[torch.Tensor, np.ndarray],
            ) -> torch.Tensor:

        s = torch.Tensor(s).to(self.device).unsqueeze(0)
        _, ActionValue = self.model.forward(s)
        ActionValue = ActionValue.squeeze(0)

        return ActionValue

    # In Reinforcement learning,
    # pi means the function from state space to action probability distribution
    # Returns probability of taken action a from state s
    def pi(
            self,
            s: torch.Tensor,
            a: torch.Tensor) -> torch.Tensor:

        a = a.unsqueeze(dim=-1)

        _, actionValue = self.model.forward(s)

        return self.policy.pi(
                actionValue,
                s,
                a)

    def log_prob(
            self,
            s: torch.Tensor,
            a: torch.Tensor) -> torch.Tensor:

        a = a.unsqueeze(dim=-1)

        _, actionValue = self.model.forward(s)

        return self.policy.log_prob(
                actionValue,
                s,
                a)

    # Get Action from State s
    @torch.no_grad()
    def get_action(
            self,
            s: Union[torch.Tensor, np.ndarray],
            isTest: bool = False,
            ) -> torch.Tensor:

        ActionValue = self.action_value(s)

        return self.policy(
                ActionValue,
                self.stepsDone,
                isTest=isTest,
                )
