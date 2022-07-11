
import random 
from collections import namedtuple, deque
from abc import abstractmethod

# PyTorch
import torch
import torch.nn as nn

from module.RL import RL

class ActorCritic(RL):

    '''
    params_dict = {
        'device': device to use, 'cuda' or 'cpu'
        'env': environment like gym
        'model': torch models for policy and value funciton
        'optimizer': torch optimizer
        'maxTimesteps': maximum timesteps agent take 
        'discount_rate': GAMMA # step-size for updating Q value
        'eps': {
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
    ):

        # init parameters 
        super().__init__(
            device = device,
            env = env,
            model = model,
            optimizer = optimizer,
            eps = eps,
            isRender = isRender,
            useTensorboard = useTensorboard,
            tensorboardParams = tensorboardParams,
            policy = policy
        )
        
        self.maxTimesteps = maxTimesteps
        self.discount_rate = discount_rate
        self.steps_done = 0 # for epsilon scheduling
        
        # Stochastic action selection
        self.softmax = nn.Softmax(dim=0)

        # torch.log makes nan(not a number) error, so we have to add some small number in log function
        self.ups=1e-7

    def __get_eps(self):
        import math

        eps_start = self.eps['start']
        eps_end = self.eps['end']
        eps_decay = self.eps['decay']

        return eps_end + (eps_start + eps_end) * math.exp(-1. * self.steps_done / eps_decay)

    # In Reinforcement learning, pi means the function from state space to action probability distribution
    # Returns probability of taken action a from state s
    def pi(self, s, a):
        s = torch.Tensor(s).to(self.device)
        _, probs = self.model.forward(s)
        probs = torch.squeeze(probs, 0)
        return probs[a]
    
    # Returns the action from state s by using multinomial distribution
    @abstractmethod
    def get_action(self, s, useEps, useStochastic):
        with torch.no_grad():
            s = torch.tensor(s).to(self.device)
            _, probs = self.model.forward(s)
            probs = torch.squeeze(probs, 0)

            eps = self.__get_eps() if useEps else 0
            
            if random.random() >= eps:
                if useStochastic:
                    probs = self.softmax(probs)

                    a = probs.multinomial(num_samples=1) 
                    a = a.data
                    action = a[0]
                else:
                    action = torch.argmax(probs, dim=0)
            else:
                a = torch.rand(probs.shape).multinomial(num_samples=1)
                a = a.data
                action = a[0]

            return action
  
    # Returns a value of the state (state value function in Reinforcement learning)
    def value(self, s):
        s = torch.tensor(s).to(self.device)
        value, _ = self.model.forward(s)
        value = torch.squeeze(value, 0)

        return value    

