
import numpy as np
import random 
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from copy import deepcopy
from abc import abstractmethod

# PyTorch
import torch
import torch.nn as nn

from module.RL import RL

class ValueBased(RL):

    '''
    param_dict = {
        'device': device to use, 'cuda' or 'cpu'
        'env':  environment like gym
        'model': torch models for policy and value funciton
        'optimizer': torch optimizer
        'maxTimesteps': maximum timesteps agent take 
        'discount_rate': step-size for updating Q value
        'maxMemory': capacitiy of Replay Memory
        'numBatch': number of batches
        'eps': { # for epsilon scheduling
            'start': 0.9,
            'end': 0.05,
            'decay': 200
        },
        'trainPolicy': select from greedy, eps-greedy, stochastic, eps-stochastic
        'testPolicy': select from greedy, eps-greedy, stochastic, eps-stochastic
    }
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
        discount_rate,
        numBatch,
        eps,
        isRender,
        useTensorboard,
        tensorboardParams,
        policy
    ):

        # init parameters 
        super().__init__(
            device=device,
            trainEnv=trainEnv,
            testEnv=testEnv,
            env=env,
            model=model,
            optimizer=optimizer,
            eps=eps,
            policy=policy,
            isRender=isRender,
            useTensorboard=useTensorboard,
            tensorboardParams=tensorboardParams 
        )
        
        self.maxTimesteps = maxTimesteps 
        self.maxMemory = maxMemory
        self.discount_rate = discount_rate
        self.numBatch = numBatch
        self.steps_done = 0 # eps scheduling
        
        # Stochastic action selection
        self.softmax = nn.Softmax(dim=0)
        
        # torch.log makes nan(not a number) error, so we have to add some small number in log function
        self.ups=1e-7

    # In Reinforcement learning, pi means the function from state space to action probability distribution
    # Returns probability of taken action a from state s
    def pi(self, s, a):
        value = self.value(s)
        a = torch.tensor(a).to(self.device).unsqueeze(axis=-1)
        actionValue = torch.gather(torch.clone(value), 1, a).squeeze(axis=1)

        return actionValue
    
    # Epsilon scheduling
    def __get_eps(self):
        import math

        eps_start = self.eps['start']
        eps_end = self.eps['end']
        eps_decay = self.eps['decay']

        eps_threshold = eps_end + (eps_start - eps_end) * math.exp(-1. * self.steps_done / eps_decay)

        return eps_threshold

    # Returns the action from state s by using multinomial distribution
    @abstractmethod
    @torch.no_grad()
    def get_action(self, s, useEps, useStochastic):
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

        return action

    def max_value(self, s):
        value = self.value(s)

        with torch.no_grad():
            maxValue = torch.max(torch.clone(value), dim=1).values
        
        return maxValue
