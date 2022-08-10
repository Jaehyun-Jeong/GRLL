
import numpy as np
import random 
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from copy import deepcopy
from abc import abstractmethod

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torch.optim as optim
from torch.autograd import Variable

from module.ValueBased.ValueBased import ValueBased

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

class ADQN(ValueBased):

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
        maxMemory: Memory size for Experience Replay
        numBatch: Batch size for mini-batch gradient descent
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
        numPrevModels: ADQN averages last k models, this parameter determines how many models it save
    '''

    def __init__(
        self, 
        model,
        optimizer,
        trainEnv=None,
        testEnv=None,
        env=None,
        device=torch.device('cpu'),
        eps={
            'start': 0.99,
            'end': 0.0001,
            'decay': 10000
        },
        maxTimesteps=1000,
        discount_rate=0.99,
        maxMemory=10000,
        numBatch=64,
        isRender={
            'train': False,
            'test': False
        }, 
        useTensorboard=False,
        tensorboardParams={
            'logdir': "./runs",
            'tag': "DQN",
        },
        policy={
            'train': 'stochastic',
            'test': 'greedy'
        },
        verbose=1,
        numPrevModels=10,
    ):

        # init parameters 
        super().__init__(
            trainEnv,
            testEnv,
            env,
            model=model,
            optimizer=optimizer,
            device=device,
            maxTimesteps=maxTimesteps,
            maxMemory=maxMemory,
            discount_rate=discount_rate,
            numBatch=numBatch,
            eps=eps,
            isRender=isRender,
            useTensorboard=useTensorboard,
            tensorboardParams=tensorboardParams,
            policy=policy,
            verbose=verbose,
        )
        
        self.replayMemory = ReplayMemory(maxMemory)

        # save last K previously learned Q-networks
        self.prevModels = deque([], maxlen=numPrevModels)

        self.printInit()

    # action seleted from previous K models by averaging it
    @abstractmethod
    def value(self, s):
        s = torch.Tensor(s).to(self.device)

        values = self.model.forward(s)
        for model in list(self.prevModels)[:-1]: # last model is equal to self.model
            values += model.forward(s)
        
        values = values / len(self.prevModels)

        return values

    # Update weights by using Actor Critic Method
    def update_weight(self):
        batch_size = self.numBatch if self.numBatch <= self.replayMemory.memory.__len__() else self.replayMemory.__len__() # if memory is smaller then numBatch, then just use all data
        batches = self.replayMemory.sample(batch_size)
        lenLoss = len(batches)

        S_t = [transition.state for transition in batches]
        A_t = [transition.action for transition in batches]
        done = [transition.done for transition in batches]
        S_tt = [transition.next_state for transition in batches]
        R_tt = [transition.reward for transition in batches]

        S_t = np.array(S_t)
        A_t = np.array(A_t)
        done = np.array(done)
        notDone = torch.Tensor(~done).to(self.device)
        S_tt = np.array(S_tt)
        R_tt = torch.Tensor(np.array(R_tt)).to(self.device)

        actionValue = self.pi(S_t, A_t)
        nextMaxValue = self.max_value(S_tt)

        target = Variable(R_tt + self.discount_rate * nextMaxValue * notDone)
        loss = 1/2 * (target - actionValue).pow(2)
        loss = torch.sum(loss)/lenLoss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.steps_done += 1

    def train(
        self, 
        maxEpisodes, 
        testPer=10, 
        testSize=10,
    ):
        try:
            returns = []
            self.prevModels.append(deepcopy(self.model)) # save initial model

            for i_episode in range(maxEpisodes):

                state = self.trainEnv.reset()
                done = False
                self.trainedEpisodes += 1
                
                #==========================================================================
                # MAKE TRAIN DATA
                #==========================================================================

                # while not done:
                for timesteps in range(self.maxTimesteps):
                    self.trainedTimesteps += 1

                    if self.isRender['train']:
                       self.trainEnv.render()

                    action = self.get_action(state, useEps=self.useTrainEps, useStochastic=self.useTrainStochastic)
                    next_state, reward, done, _ = self.trainEnv.step(action.tolist())
                    self.replayMemory.push(state, action, done, next_state, reward)
                    state = next_state

                    if done or timesteps == self.maxTimesteps-1:
                        break

                    # train
                    self.update_weight()
                    self.prevModels.append(deepcopy(self.model)) # save updated model

                #==========================================================================
                # TEST
                #==========================================================================
                if self.trainedEpisodes % testPer == 0: 

                    cumulative_rewards = self.test(testSize=testSize)
                    returns.append(cumulative_rewards)

                    #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
                    # TENSORBOARD

                    self.writeTensorboard(returns[-1], self.trainedEpisodes)

                    #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

                    self.printResult(self.trainedEpisodes, self.trainedTimesteps, returns[-1])

        except KeyboardInterrupt:
            print("KeyboardInterruption occured")

            plt.plot(range(len(returns)), returns)
        finally:
            plt.plot(range(len(returns)), returns)

        self.trainEnv.close()
