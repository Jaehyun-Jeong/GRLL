
import numpy as np
import random 
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from copy import deepcopy

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torch.optim as optim
from torch.autograd import Variable

Transition = namedtuple('Transition',
                       ('state', 'action', 'done', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class ADQN():

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
        env, 
        model, 
        optimizer, 
        maxTimesteps=1000, 
        maxMemory=10000, 
        eps={
            'start': 0.9,
            'end': 0.05,
            'decay': 200
        }, 
        device="cpu", 
        discount_rate=0.99, 
        numBatch=64, 
        trainPolicy='eps-greedy',
        testPolicy='greedy'
    ):

        # init parameters 
        self.device = device
        self.env = env
        self.model = model
        self.optimizer = optimizer
        self.maxTimesteps = maxTimesteps 
        self.discount_rate = discount_rate
        self.replayMemory = ReplayMemory(maxMemory)
        self.numBatch = numBatch
        self.eps = eps
        self.steps_done = 0 # eps scheduling
        
        # Stochastic action selection
        self.softmax = nn.Softmax(dim=0)

        # select train, test policy
        policyDict = {'greedy': [False, False], 'stochastic': [False, True], 'eps-greedy': [True, False], 'eps-stochastic': [True, True]} # [ useEpsilon, useStochastic ]

        try:
            trainPolicyList = policyDict[trainPolicy]
            testPolicyList = policyDict[testPolicy]

            if trainPolicyList[0] or testPolicyList[0]:
                self.eps = eps

            self.useTrainEps = trainPolicyList[0]
            self.useTrainStochastic = trainPolicyList[1]
            self.useTestEps = testPolicyList[0]
            self.useTestStochastic = testPolicyList[1]

        except: 
            print("ERROR OCCURED : supported policies are 'greedy', 'eps-greedy', 'stochastic', and 'eps-stochastic'")
        
        # torch.log makes nan(not a number) error, so we have to add some small number in log function
        self.ups=1e-7

    # In Reinforcement learning, pi means the function from state space to action probability distribution
    # Returns probability of taken action a from state s
    def pi(self, s, a):
        s = torch.Tensor(s).to(self.device)
        value = self.model.forward(s)
        value = torch.squeeze(value, 0)
        return value[a]
    
    # Epsilon scheduling
    def __get_eps(self):
        import math

        eps_start = self.eps['start']
        eps_end = self.eps['end']
        eps_decay = self.eps['decay']

        eps_threshold = eps_end + (eps_start - eps_end) * math.exp(-1. * self.steps_done / eps_decay)

        return eps_threshold

    # Returns the action from state s by using multinomial distribution
    def get_action(self, values, useEps, useStochastic):
        with torch.no_grad():
            s = torch.tensor(s).to(self.device)
            probs = self.value(s)

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

    # action seleted from previous K models by averaging it
    def averaged_value(self, s):
        with torch.no_grad():

            prevModels = list(self.prevModels)

            values = self.model.forward(s)
            for model in prevModels[:-1]: # last model is equal to self.model
                values += self.model.forward(s)
            
            values = values / len(self.prevModels)
            values = torch.squeeze(values, 0)

            return values


    # Returns a value of the state (state value function in Reinforcement learning)
    def max_value(self, s):
        with torch.no_grad():

            s = torch.tensor(s).to(self.device)
            values = self.averaged_value(s)
            maxValues = torch.max(values)

            return maxValues
    
    def test(self, isRender=True, testSize=10):
        
        returns = []

        for testIdx in range(testSize):
            state = self.env.reset()
            done = False
            rewards = []
            for timesteps in range(self.maxTimesteps):
                if isRender:
                    self.env.render()

                action = self.get_action(state, useEps=self.useTestEps, useStochastic=self.useTestStochastic)
                next_state, reward, done, _ = self.env.step(action.tolist())

                rewards.append(reward)
                state = next_state

                if done or timesteps == self.maxTimesteps-1:
                    break
            
            returns.append(sum(rewards))
        
        averagedReward = sum(returns) / testSize

        self.env.close()

        return averagedReward
