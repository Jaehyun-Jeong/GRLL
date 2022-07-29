
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

# Parent Class
from module.PolicyGradient import PolicyGradient

Transition = namedtuple('Transition',
                       ('state', 'action', 'next_state', 'reward'))

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

class REINFORCE(PolicyGradient):

    '''
    params_dict = {
        'device': device to use, 'cuda' or 'cpu'
        'env': environment like gym
        'model': torch models for policy and value funciton
        'optimizer': torch optimizer
        #MAX_EPISODES = maximum episodes you want to learn
        'maxTimesteps': maximum timesteps agent take 
        'discount_rate': GAMMA # step-size for updating Q value
        'epsilon': epsilon for epsilon greedy action
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
        isRender={
            'train': False,
            'test': False,
        },
        useTensorboard=False,
        tensorboardParams={
            'logdir': "./runs/REINFORCE",
            'tag': "Returns"
        },
        policy={
            'train': 'eps-stochastic',
            'test': 'stochastic'
        },
        useBaseline=True,
    ):

        # init parameters 
        super().__init__(
            env=env,
            model=model,
            optimizer=optimizer,
            device=device,
            maxTimesteps=maxTimesteps,
            discount_rate=discount_rate,
            eps=eps,
            isRender=isRender,
            useTensorboard=useTensorboard,
            tensorboardParams=tensorboardParams,
            policy=policy
        )
        
        self.useBaseline=useBaseline
        
        # torch.log makes nan(not a number) error, so we have to add some small number in log function
        self.ups=1e-7

    # Update weights by using Actor Critic Method
    def __update_weight(self, Transitions, entropy_term = 0):
        Qval = 0
        loss = 0
        lenLoss = Transitions.memory.__len__()

        # update by using mini-batch Gradient Ascent
        for Transition in reversed(Transitions.memory):
            s_t = Transition.state
            a_t = Transition.action
            s_tt = Transition.next_state
            r_tt = Transition.reward

            Qval = r_tt + self.discount_rate * Qval

            # get actor loss
            log_prob = torch.log(self.pi(s_t, a_t) + self.ups)
            advantage = Variable(Qval - self.value(s_t) * self.useBaseline)
            actor_loss = -(advantage * log_prob)

            # get critic loss
            value = self.value(s_t)
            critic_loss = 1/2 * (Qval - value).pow(2)

            loss += actor_loss + critic_loss + 0.001 * entropy_term

        loss = loss/lenLoss
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train(
        self, 
        maxEpisodes, 
        testPer=10, 
        testSize=10,
    ):

        try:
            returns = []
            
            for i_episode in range(maxEpisodes):

                Transitions = ReplayMemory(maxEpisodes)
                state = self.env.reset()
                done = False
                self.trainedEpisodes += 1
                
                #==========================================================================
                # MAKE TRAIN DATA
                #==========================================================================

                # while not done:
                for timesteps in range(self.maxTimesteps):
                    self.trainedTimesteps += 1

                    if self.isRender['train']:
                        env.render()

                    action = self.get_action(state, useEps=self.useTrainEps, useStochastic=self.useTrainStochastic)
                    next_state, reward, done, _ = self.env.step(action.tolist())
                    Transitions.push(state, action, next_state, reward)
                    state = next_state

                    if done or timesteps == self.maxTimesteps-1:
                        break
                # train
                self.__update_weight(Transitions)

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

    ALPHA = 0.1e-3 # learning rate
    GAMMA = 0.99 # discount_rate
    epsilon = 0 # for epsilon greedy action

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

    REINFORCE_parameters = {
        'device': device, # device to use, 'cuda' or 'cpu'
        'env': env, # environment like gym
        'model': ACmodel, # torch models for policy and value funciton
        'optimizer': optimizer, # torch optimizer
        'maxTimesteps': MAX_TIMESTEPS, # maximum timesteps agent take 
        'discount_rate': GAMMA, # step-size for updating Q value
        'epsilon': epsilon, # epsilon greedy action for training
        'useBaseline': True # use value function as baseline or not
    }

    # Initialize REINFORCE Mehtod
    RF = REINFORCE(**REINFORCE_parameters)

    # TRAIN Agent
    RF.train(MAX_EPISODES, isRender=False, useTensorboard=True, tensorboardTag="CartPole-v1")

