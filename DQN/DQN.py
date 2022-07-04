
import numpy as np
import random 
import matplotlib.pyplot as plt
from collections import namedtuple, deque

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

class DQN():

    '''
    param_dict = {
        'device': device to use, 'cuda' or 'cpu'
        'env':  environment like gym
        'model': torch models for policy and value funciton
        'optimizer': torch optimizer
        'maxTimesteps': maximum timesteps agent take 
        'discount_rate': step-size for updating Q value
        'epsilon': epsilon greedy action for training
        'maxMemory': capacitiy of Replay Memory
        'numBatch': number of batches
    }
    '''

    def __init__(self, **params_dict):
        super(DQN, self).__init__()

        # init parameters 
        self.device = params_dict['device']
        self.env = params_dict['env']
        self.model = params_dict['model']
        self.optimizer = params_dict['optimizer']
        self.maxTimesteps = params_dict['maxTimesteps'] 
        self.discount_rate = params_dict['discount_rate']
        self.epsilon = params_dict['epsilon']
        self.replayMemory = ReplayMemory(params_dict['maxMemory'])
        self.numBatch = params_dict['numBatch']

        # torch.log makes nan(not a number) error, so we have to add some small number in log function
        self.ups=1e-7

    # In Reinforcement learning, pi means the function from state space to action probability distribution
    # Returns probability of taken action a from state s
    def pi(self, s, a):
        s = torch.Tensor(s).to(self.device)
        value = self.model.forward(s)
        value = torch.squeeze(value, 0)
        return value[a]
    
    # Returns the action from state s by using multinomial distribution
    def get_action(self, s, epsilon = 0): # epsilon 0 for greedy action
        with torch.no_grad():
            s = torch.tensor(s).to(self.device)
            values = self.model.forward(s)
            probs = self.model.softmax(values)
            probs = torch.squeeze(probs, 0)

            if random.random() >= epsilon:
                action = torch.argmax(probs, dim=0)
            else:
                a = torch.rand(probs.shape).multinomial(num_samples=1)
                a = a.data
                action = a[0]

            return action
  
    # Returns a value of the state (state value function in Reinforcement learning)
    def max_value(self, s):
        s = torch.tensor(s).to(self.device)
        value = self.model.forward(s)
        value = torch.squeeze(value, 0)
        maxValue = torch.max(value)

        return maxValue

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

    def test(self, isRender=False):
        state = self.env.reset()
        done = False
        rewards = []

        for timesteps in range(self.maxTimesteps):
            if isRender:
                self.env.render()

            action = self.get_action(state)
            next_state, reward, done, _ = self.env.step(action.tolist())

            rewards.append(reward)
            state = next_state

            if done or timesteps == self.maxTimesteps-1:
                break

        return sum(rewards)

    def train(self, maxEpisodes, testPer=10, isRender=False, useTensorboard=False, tensorboardTag="DQN"):
        try:
            returns = []

            #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
            # TENSORBOARD
            
            if useTensorboard:
                from torch.utils.tensorboard import SummaryWriter
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

                    action = self.get_action(state, epsilon=self.epsilon)
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
    epsilon = 0.3 # for epsilon greedy action

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
        'epsilon': epsilon, # epsilon greedy action for training
        'maxMemory': MAX_REPLAYMEMORY, # capacitiy of Replay Memory
        'numBatch': 64 # number of batches
    }

    # Initialize DQN Mehtod
    DeepQN = DQN(**param_dict)

    # TRAIN Agent
    DeepQN.train(MAX_EPISODES, isRender=False, useTensorboard=True, tensorboardTag="CartPole-v1")

