
from collections import namedtuple, deque
from abc import abstractmethod
import random

# PyTorch
import torch
import torch.nn as nn

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
            'train': If it's True, then render environment screen while training, or vice versa
            'test': If it's True, then render environment screen while testing, or vice versa
        }
        useTensorboard: False means not using TensorBaord
        tensorboardParams={ TensorBoard parameters
            'logdir': Saved directory
            'tag':
        }
        policy={ there are 4 types of Policy 'stochastic', 'eps-stochastic', 'greedy', 'eps-greedy'
            'train': e.g. 'eps-stochastic'
            'test': e.g. 'stochastic'
        }
        verbose: The verbosity level: 0 no output, 1 only train info, 2 train info + initialized info
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
            eps=eps,
            isRender=isRender,
            useTensorboard=useTensorboard,
            tensorboardParams=tensorboardParams,
            policy=policy,
            verbose=verbose,
        )
        
        self.maxTimesteps = maxTimesteps
        self.discount_rate = discount_rate
        self.steps_done = 0 # for epsilon scheduling
        
        # Stochastic action selection
        self.softmax = nn.Softmax(dim=0)

        # torch.log makes nan(not a number) error, so we have to add some small number in log function
        self.ups=1e-7

    # Epsilon scheduling method
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
        a = torch.tensor(a).to(self.device).unsqueeze(axis=-1)

        _, probs = self.model.forward(s)
        actionValue = torch.gather(torch.clone(probs), 1, a).squeeze(axis=1)

        return actionValue
    
    # Returns the action from state s by using multinomial distribution
    @abstractmethod
    @torch.no_grad()
    def get_action(self, s, useEps, useStochastic):
        s = torch.Tensor(s).to(self.device)
        _, probs = self.model.forward(s)
        probs = torch.squeeze(probs, 0)

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

        return action.detach()
  
    # Returns a value of the state (state value function in Reinforcement learning)
    def value(self, s):
        s = torch.Tensor(s).to(self.device)
        value, _ = self.model.forward(s)
        value = value.squeeze(-1)

        return value
