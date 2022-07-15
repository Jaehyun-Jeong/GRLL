
import numpy as np
import random 
import matplotlib.pyplot as plt
from collections import namedtuple, deque
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

class DQN(ValueBased):

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
        }
    }
    '''

    def __init__(
        self, 
        env,
        model,
        optimizer,
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
            'tag': "DQN"
        },
        policy={
            'train': 'eps-greedy',
            'test': 'stochastic'
        },
    ):

        # init parameters 
        super().__init__(
            env=env,
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
            policy=policy
        )
        
        self.replayMemory = ReplayMemory(maxMemory)

        # torch.log makes nan(not a number) error, so we have to add some small number in log function
        self.ups=1e-7
    
    # ADQN has different type of value
    @abstractmethod
    def value(self, s):
        s = torch.Tensor(s).to(self.device)
        value = self.model.forward(s)
        value = torch.squeeze(value, 0)
        return value

    # Update weights by using Actor Critic Method
    def update_weight(self):
        loss = 0
        batch_size = self.numBatch if self.numBatch <= self.replayMemory.memory.__len__() else self.replayMemory.__len__() # if memory is smaller then numBatch, then just use all data
        batches = self.replayMemory.sample(batch_size)
        lenLoss = len(batches)

        # update by using mini-batch Gradient Descent
        for Transition in batches:
            s_t = Transition.state
            a_t = Transition.action
            done = Transition.done
            s_tt = Transition.next_state
            r_tt = Transition.reward
            
            target = Variable(r_tt + self.discount_rate * self.max_value(s_tt) * (not done))
            loss += 1/2 * (target - self.pi(s_t, a_t)).pow(2)

        loss = loss/lenLoss
        
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

            for i_episode in range(maxEpisodes):

                state = self.env.reset()
                done = False
                
                #==========================================================================
                # MAKE TRAIN DATA
                #==========================================================================

                # while not done:
                for timesteps in range(self.maxTimesteps):

                    if self.isRender['train']:
                       self.env.render()

                    action = self.get_action(state, useEps=self.useTrainEps, useStochastic=self.useTrainStochastic)
                    next_state, reward, done, _ = self.env.step(action.tolist())
                    self.replayMemory.push(state, action, done, next_state, reward)
                    state = next_state

                    if done or timesteps == self.maxTimesteps-1:
                        break

                    # train
                    self.update_weight()

                #==========================================================================
                # TEST
                #==========================================================================
                if (i_episode+1) % testPer == 0: 

                    cumulative_rewards = self.test(testSize=testSize)
                    returns.append(cumulative_rewards)

                    #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
                    # TENSORBOARD

                    self.writeTensorboard(returns[-1], i_episode+1)

                    #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
                    
                    self.printResult(i_episode + 1, returns[-1])

        except KeyboardInterrupt:
            print("==============================================")
            print("KEYBOARD INTERRUPTION!!=======================")
            print("==============================================")

            plt.plot(range(len(returns)), returns)
        finally:
            plt.plot(range(len(returns)), returns)

        self.env.close()

if __name__ == "__main__":

    from models import ANN_V1 # import model
    import gym # Environment 

    MAX_EPISODES = 10000
    MAX_TIMESTEPS = 1000
    MAX_REPLAYMEMORY = 10000

    ALPHA = 0.1e-3 # learning rate
    GAMMA = 0.99 # discount_rate

    # device to use
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # set environment
    env = gym.make("CartPole-v0")
    #env = gym.make("Acrobot-v1")
    #env = gym.make("MountainCar-v0")

    num_actions = env.action_space.n
    num_states = env.observation_space.shape[0]

    ACmodel = ANN_V1(num_states, num_actions).to(device)
    optimizer = optim.Adam(ACmodel.parameters(), lr=ALPHA)

    param_dict = {
        'device': device, # device to use, 'cuda' or 'cpu'
        'env': env, # environment like gym
        'model': ACmodel, # torch models for policy and value funciton
        'optimizer': optimizer, # torch optimizer
        'maxTimesteps': MAX_TIMESTEPS, # maximum timesteps agent take 
        'discount_rate': GAMMA, # step-size for updating Q value
        'maxMemory': MAX_REPLAYMEMORY, # capacitiy of Replay Memory
        'numBatch': 64, # number of batches
        'eps': { # for epsilon scheduling
            'start': 0.9,
            'end': 0.05,
            'decay': 200
        }
    }

    # Initialize DQN Mehtod
    DeepQN = DQN(**param_dict)

    # TRAIN Agent
    DeepQN.train(MAX_EPISODES, isRender=False, useTensorboard=True, tensorboardTag="CartPole-v1")

if __name__ == "__main__":

    from models import ANN_V1 # import model
    import gym # Environment 

    MAX_EPISODES = 10000
    MAX_TIMESTEPS = 1000
    MAX_REPLAYMEMORY = 10000

    ALPHA = 0.1e-3 # learning rate
    GAMMA = 0.99 # discount_rate

    # device to use
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # set environment
    env = gym.make("CartPole-v0")
    #env = gym.make("Acrobot-v1")
    #env = gym.make("MountainCar-v0")

    num_actions = env.action_space.n
    num_states = env.observation_space.shape[0]

    ACmodel = ANN_V1(num_states, num_actions).to(device)
    optimizer = optim.Adam(ACmodel.parameters(), lr=ALPHA)

    param_dict = {
        'device': device, # device to use, 'cuda' or 'cpu'
        'env': env, # environment like gym
        'model': ACmodel, # torch models for policy and value funciton
        'optimizer': optimizer, # torch optimizer
        'maxTimesteps': MAX_TIMESTEPS, # maximum timesteps agent take 
        'discount_rate': GAMMA, # step-size for updating Q value
        'maxMemory': MAX_REPLAYMEMORY, # capacitiy of Replay Memory
        'numBatch': 64, # number of batches
        'eps': { # for epsilon scheduling
            'start': 0.9,
            'end': 0.05,
            'decay': 200
        }
    }

    # Initialize DQN Mehtod
    DeepQN = DQN(**param_dict)

    # TRAIN Agent
    DeepQN.train(MAX_EPISODES, isRender=False, useTensorboard=True, tensorboardTag="CartPole-v1")

