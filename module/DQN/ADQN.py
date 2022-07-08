
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
        'kFold': to save last K previously learnd Q-networks
    }
    '''

    def __init__(self, env, model, optimizer, maxTimesteps, maxMemory, eps, device="cpu", discount_rate=0.99, numBatch=64, numPrevModels=10):
        super(ADQN, self).__init__()

        # init parameters 
        super().__init__(
            device = device
            env = env
            model = model
            optimizer = optimizer
            maxTimesteps = maxTimesteps 
            discount_rate = discount_rate
            replayMemory = ReplayMemory(maxMemory)
            numBatch = numBatch
            eps = eps
        )

        # Stochastic action selection
        self.softmax = nn.Softmax(dim=0)

        # save last K previously learned Q-networks 
        self.kFold = numPrevModels
        self.prevModels = deque([], maxlen=numPrevModels)

        # torch.log makes nan(not a number) error, so we have to add some small number in log function
        self.ups=1e-7

    # action seleted from previous K models by averaging it
    def value(self, s):
        with torch.no_grad():

            prevModels = list(self.prevModels)

            values = self.model.forward(s)
            for model in prevModels[:-1]: # last model is equal to self.model
                values += self.model.forward(s)
            
            values = values / len(self.prevModels)
            values = torch.squeeze(values, 0)

            return values

    # Returns the action from state s by using multinomial distribution
    def get_action(self, s, useEps, useStochastic):
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

    # Returns a value of the state (state value function in Reinforcement learning)
    def max_value(self, s):
        with torch.no_grad():

            s = torch.tensor(s).to(self.device)
            values = self.value(s)
            maxValues = torch.max(values)

            return maxValues
    
    # Returns the action from state s by using multinomial distribution
    def get_action(self, s, isGreedy=False): # epsilon 0 for greedy action
        with torch.no_grad():
            s = torch.tensor(s).to(self.device)
            probs = self.value(s) 

            if random.random() >= self.get_eps() or isGreedy:
                action = torch.argmax(probs, dim=0)
            else:
                a = torch.rand(probs.shape).multinomial(num_samples=1)
                a = a.data
                action = a[0]

            return action

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

    def test(self, isRender=False):
        state = self.env.reset()
        done = False
        rewards = []

        for timesteps in range(self.maxTimesteps):
            if isRender:
                self.env.render()

            action = self.get_action(state, useEps=useTestEps, useStochastic=useTestStochastic)
            next_state, reward, done, _ = self.env.step(action.tolist())

            rewards.append(reward)
            state = next_state

            if done or timesteps == self.maxTimesteps-1:
                break

        return sum(rewards)

    def train(self, maxEpisodes, testPer=10, isRender=False, useTensorboard=False, tensorboardTag="DQN"):
        try:
            returns = []
            self.prevModels.append(deepcopy(self.model)) # save initial model

            #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
            # TENSORBOARD
            
            if useTensorboard:
                from tensorboardX import SummaryWriter
                self.writer = SummaryWriter()

            #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

            for i_episode in range(maxEpisodes):

                state = self.env.reset()
                done = False
                
                #==========================================================================
                # MAKE TRAIN DATA
                #==========================================================================

                # while not done:
                for timesteps in range(self.maxTimesteps):

                    if isRender:
                       self.env.render()

                    action = self.get_action(state)
                    next_state, reward, done, _ = self.env.step(action.tolist())
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
                if (i_episode+1) % testPer == 0: 

                    cumulative_rewards = self.test()
                    returns.append(cumulative_rewards)

                    #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
                    # TENSORBOARD

                    if useTensorboard:
                        self.writer.add_scalars("Returns", {tensorboardTag: returns[-1]}, i_episode+1)

                    #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

                    print("Episode: {0:<10} return: {1:<10}".format(i_episode + 1, returns[-1]))

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
    averagedDQN = ADQN(**param_dict)

    # TRAIN Agent
    averagedDQN.train(MAX_EPISODES, isRender=False, useTensorboard=True, tensorboardTag="CartPole-v1")

