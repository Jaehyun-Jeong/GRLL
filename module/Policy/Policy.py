from typing import Union, Dict

import random

# PyTorch
import torch
from torch.distributions import Normal

# module
from module.utils.utils import overrides
from module.utils.exploration import Epsilon, NormalNoise
from module.utils.ActionSpace import ActionSpace


class Policy():
    """
    parameters
        algorithm: Algorithm name want to use
            e.g.) epsilon-greedy, greedy
        exploring: Exploring algorithm name want to use
        exploringParams:
            Exploring parameters selected depanding exploring algorithm
            e.g.)
                When using epsilon greedy
                'exploringParams': {
                    'schedule': 'exponential',
                    'start': 0.99,
                    'end': 0.0001,
                    'decay': 10000
                }
    """

    def __init__(
            self,
            algorithm: str,  # Policy Algorithm
            exploring: str,  # Exploring Method
            exploringParams: Dict[str, Union[int, float]],
            actionSpace: ActionSpace):

        self.algorithm = algorithm
        self.exploring = exploring
        self.exploringParams = exploringParams
        self.actionSpace = actionSpace

    def __call__(self) -> torch.Tensor:
        raise NotImplementedError()

    def pi(self) -> torch.Tensor:
        raise NotImplementedError()


class DiscretePolicy(Policy):

    def __init__(
            self,
            actionSpace: ActionSpace,
            algorithm: str = "greedy",  # greedy, stochastic
            exploring: str = "epsilon",  # epsilon, None
            exploringParams: Dict[str, Union[int, float]] = {
                    'schedule': 'exponential',
                    'start': 0.99,
                    'end': 0.0001,
                    'decay': 10000
                },
            ):

        super().__init__(
                algorithm=algorithm,
                exploring=exploring,
                exploringParams=exploringParams,
                actionSpace=actionSpace,
                )

        # Initialize Parameters
        self.useEps = False
        self.useStochastic = False
        if exploring == 'epsilon':
            self.useEps = True
            self.exploration = Epsilon(
                    **exploringParams)
        if algorithm == 'stochastic':
            self.useStochastic = True

    # Return Action
    @overrides(Policy)
    def __call__(
            self,
            actionValue: torch.Tensor,
            stepsDone: int,
            ) -> list:

        probs = actionValue

        eps = self.exploration(stepsDone) if self.useEps else 0

        if random.random() >= eps:
            if self.useStochastic:
                probs = self.softmax(probs)

                a = probs.multinomial(num_samples=1)
                a = a.data
                action = a[0].cpu()
            else:
                action = torch.argmax(probs, dim=0)
        else:
            a = torch.rand(probs.shape).multinomial(num_samples=1)
            a = a.data
            action = a[0]

        action = action.detach()

        return action.tolist()

    # In Reinforcement learning,
    # pi means the function from state space to action probability distribution
    # Returns probability of taken action a from state s
    @overrides(Policy)
    def pi(
            self,
            actionValue: torch.Tensor,
            s: torch.Tensor,
            a: torch.Tensor) -> torch.Tensor:

        actionValue = torch.gather(
                torch.clone(actionValue), 1, a).squeeze(dim=1)

        return actionValue


class ContinuousPolicy(Policy):

    def __init__(
            self,
            actionSpace: ActionSpace,
            algorithm: str = "plain",  # plain
            exploring: str = "normal",  # normal, None
            exploringParams: Dict[str, Union[int, float]] = {
                    'mean': 0,  # mean
                    'sigma': 1,  # standard deviation
                },
            ):

        super().__init__(
                algorithm=algorithm,
                exploring=exploring,
                exploringParams=exploringParams,
                actionSpace=actionSpace,
                )

        # Initialize Parameters
        if exploring == 'normal':
            self.exploration = NormalNoise(
                    **exploringParams)

    # Return Action
    @overrides(Policy)
    def __call__(
            self,
            actionValue: torch.Tensor,
            stepsDone: int,
            ) -> list:

        # Get noise
        noise = self.exploration(
                stepsDone,
                actionValue.shape)

        # Add noise to action
        action = actionValue + noise

        # clamp action by high and low
        action = torch.clamp(
                action,
                min=torch.Tensor(self.actionSpace.low),
                max=torch.Tensor(self.actionSpace.high))

        return action.tolist()

    # In Reinforcement learning,
    # pi means the function from state space to action probability distribution
    # Returns probability of taken action a from state s
    @overrides(Policy)
    def pi(
            self,
            actionValue: torch.Tensor,
            s: torch.Tensor,
            a: torch.Tensor) -> torch.Tensor:

        a = a.squeeze(dim=-1)
        dist = Normal(actionValue, torch.ones(actionValue.shape))
        logProb = dist.log_prob(a).sum(dim=1)

        return dist.log_prob(a)
