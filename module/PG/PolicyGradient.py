from typing import Union

import random
import numpy as np

# PyTorch
import torch
import torch.nn as nn

from module.utils import overrides
from module.RL import RL


class PolicyGradient(RL):

    '''
    parameters
        model: torch.nn.Module based model for state_value, and action_value
        optimizer: torch optimizer
        trainEnv: Environment which is used to train
        testEnv: Environment which is used to test
        env: only for when it don't need to be split by trainEnv, testEnv
        device: Device used for training, like Backpropagation
        eps={
            'start': Start epsilon value for epsilon greedy policy
            'end': Final epsilon value for epsilon greedy policy
            'decay': It determines how small epsilon is
        }
        maxTimesteps: Permitted timesteps in the environment
        discount_rate: Discount rate for calculating return(accumulated reward)
        isRender={

            'train':
            If it's True,
            then render environment screen while training, or vice versa

            'test':
            If it's True,
            then render environment screen while testing, or vice versa

        }
        useTensorboard: False means not using TensorBaord
        tensorboardParams={ TensorBoard parameters
            'logdir': Saved directory
            'tag':
        }
        policy={

            there are 4 types of Policy
            'stochastic',
            'eps-stochastic',
            'greedy',
            'eps-greedy'

            'train': e.g. 'eps-stochastic'
            'test': e.g. 'stochastic'
        }
        verbose: The verbosity level:
            0 no output,
            1 only train info,
            2 train info + initialized info
    '''

    def __init__(
        self,
        trainEnv,
        testEnv,
        env,
        model,
        optimizer,
        device,
        maxTimesteps,
        discount_rate,
        eps,
        isRender,
        useTensorboard,
        tensorboardParams,
        policy,
        verbose,
    ):

        # init parameters
        super().__init__(
            device=device,
            trainEnv=trainEnv,
            testEnv=testEnv,
            env=env,
            model=model,
            optimizer=optimizer,
            maxTimesteps=maxTimesteps,
            eps=eps,
            isRender=isRender,
            useTensorboard=useTensorboard,
            tensorboardParams=tensorboardParams,
            policy=policy,
            verbose=verbose,
        )

        self.discount_rate = discount_rate
        self.steps_done = 0  # for epsilon scheduling

        # Stochastic action selection
        self.softmax = nn.Softmax(dim=0)

        # torch.log makes nan(not a number) error,
        # so we have to add some small number in log function
        self.ups = 1e-7

    # Epsilon scheduling method
    def __get_eps(self) -> float:
        import math

        eps_start = self.eps['start']
        eps_end = self.eps['end']
        eps_decay = self.eps['decay']

        return eps_end + \
            (eps_start + eps_end) * math.exp(-1. * self.steps_done / eps_decay)

    # Returns the action from state s by using multinomial distribution
    @overrides(RL)
    @torch.no_grad()
    def get_action(
            self,
            s: Union[torch.Tensor, np.ndarray],
            useEps: bool,
            useStochastic: bool) -> torch.Tensor:

        s = torch.Tensor(s).to(self.device).unsqueeze(0)
        _, probs = self.model.forward(s)
        probs = probs.squeeze(0)

        eps = self.__get_eps() if useEps else 0

        if random.random() >= eps:
            if useStochastic:
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

    # Returns a value of the state
    # (state value function in Reinforcement learning)
    def value(self, s: torch.Tensor) -> torch.Tensor:
        value, _ = self.model.forward(s)

        return value

    # In Reinforcement learning,
    # pi means the function from state space to action probability distribution
    # Returns probability of taken action a from state s
    def pi(
            self,
            s: torch.Tensor,
            a: torch.Tensor) -> torch.Tensor:

        a = a.unsqueeze(dim=-1)

        _, probs = self.model.forward(s)
        actionValue = torch.gather(torch.clone(probs), 1, a).squeeze(dim=1)

        return actionValue
