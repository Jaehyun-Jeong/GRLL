from typing import Union

import random
import numpy as np

# PyTorch
import torch
import torch.nn as nn

from module.RL import RL
from module.utils import overrides


class ValueBased(RL):

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
        discount: Discount rate for calculating return(accumulated reward)
        maxMemory: Memory size for Experience Replay
        numBatch: Batch size for mini-batch gradient descent
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
        gradientStepPer:
            Update the neural network model every gradientStepPer timesteps
        epoch: Epoch size to train from given data(Replay Memory)
        trainStarts:
            how many steps of the model
            to collect transitions for before learning starts
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
        maxMemory,
        discount,
        numBatch,
        eps,
        isRender,
        useTensorboard,
        tensorboardParams,
        policy,
        clippingParams,
        verbose,
        gradientStepPer,
        epoch,
        trainStarts,
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
            policy=policy,
            clippingParams=clippingParams,
            isRender=isRender,
            useTensorboard=useTensorboard,
            tensorboardParams=tensorboardParams,
            verbose=verbose
        )

        self.maxMemory = maxMemory
        self.discount = discount
        self.numBatch = numBatch
        self.gradientStepPer = gradientStepPer
        self.epoch = epoch
        self.trainStarts = trainStarts
        self.steps_done = 0  # eps scheduling

        # Stochastic action selection
        self.softmax = nn.Softmax(dim=0)

        # torch.log makes nan(not a number) error,
        # so we have to add some small number in log function
        self.ups = 1e-7

    def is_trainable(self):

        # check train condition
        condition = True if self.trainedTimesteps % self.gradientStepPer == 0 \
                    and self.trainedTimesteps >= self.trainStarts \
                    else False

        return condition

    # Value function shuld be overrided
    def value(
            self,
            s: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        pass

    # In Reinforcement learning,
    # pi means the function from state space to action probability distribution
    # Returns probability of taken action a from state s
    def pi(
            self,
            s: Union[torch.Tensor, np.ndarray],
            a: Union[torch.Tensor, np.ndarray]):

        value = self.value(s)
        a = torch.tensor(a).to(self.device).unsqueeze(dim=-1)
        actionValue = torch.gather(torch.clone(value), 1, a).squeeze(dim=1)

        return actionValue

    # Epsilon scheduling
    def __get_eps(self):
        import math

        eps_start = self.eps['start']
        eps_end = self.eps['end']
        eps_decay = self.eps['decay']

        eps_threshold = \
            eps_end + (eps_start - eps_end) * \
            math.exp(-1. * self.steps_done / eps_decay)

        return eps_threshold

    # Returns the action from state s by using multinomial distribution
    @overrides(RL)
    @torch.no_grad()
    def get_action(
            self,
            s: Union[torch.Tensor, np.ndarray],
            useEps: bool,
            useStochastic: bool):

        s = torch.Tensor(s).to(self.device).unsqueeze(0)
        probs = self.model.forward(s).squeeze(0)

        eps = self.__get_eps() if useEps else 0

        if random.random() >= eps:
            if useStochastic:
                probs = self.softmax(probs)
                a = probs.multinomial(num_samples=1)
                a = a.data
                action = a[0]
            else:
                # all actions must be in cpu, but all states in gpu if it using
                action = torch.argmax(probs, dim=0).cpu()
        else:
            a = torch.rand(probs.shape).multinomial(num_samples=1)
            a = a.data
            action = a[0]

        return action.tolist()

    # get max Q-value
    def max_value(
            self,
            s: Union[torch.Tensor, np.ndarray]):

        value = self.value(s)

        with torch.no_grad():
            maxValue = torch.max(torch.clone(value), dim=1).values

        return maxValue
