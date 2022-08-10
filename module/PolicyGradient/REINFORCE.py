
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
from module.PolicyGradient.PolicyGradient import PolicyGradient

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
        useBaseline: True means use baseline term for REINFORCE algorithm
        verbose: The verbosity level: 0 no output, 1 only train info, 2 train info + initialized info
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
        verbose=1,
        useBaseline=True,
    ):

        # init parameters 
        super().__init__(
            trainEnv=trainEnv,
            testEnv=testEnv,
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
            policy=policy,
            verbose=verbose,
        )
        
        self.useBaseline=useBaseline

        self.printInit()

    # Update weights by using Actor Critic Method
    def __update_weight(self, Transitions, entropy_term = 0):

        lenLoss = Transitions.memory.__len__()

        S_t = [transition.state for transition in Transitions.memory]
        A_t = [transition.action for transition in Transitions.memory]
        R_tt = [transition.reward for transition in Transitions.memory]

        S_t = np.array(S_t)
        A_t = np.array(A_t)

        # calculate Q-value
        Qval = [R_tt[-1]]
        for r_tt in R_tt[:-1]:
            Qval = [r_tt + self.discount_rate * Qval[0]] + Qval
        Qval = torch.Tensor(Qval).to(self.device)

        # get actor loss
        log_prob = torch.log(self.pi(S_t, A_t) + self.ups)
        advantage = Variable(Qval - self.value(S_t) * self.useBaseline)
        actor_loss = -(advantage * log_prob)

        # get critic loss
        value = self.value(S_t)
        critic_loss = 1/2 * (Qval - value).pow(2)

        # calculate loss
        loss = actor_loss + critic_loss + 0.001 * entropy_term
        loss = torch.sum(loss)/lenLoss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.steps_done += 1

    def train(
        self, 
        trainTimesteps, # Training Timesteps
        testPer=1000, # Test per testPer timesteps
        testSize=1000, # The Timesteps to test, trainTimesteps doesn't include it
    ):

        try:
            rewards = []
            
            while trainTimesteps > self.trainedTimesteps:

                Transitions = ReplayMemory(self.maxTimesteps)
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
                    Transitions.push(state, action, next_state, reward)
                    state = next_state

                    #==========================================================================
                    # TEST
                    #==========================================================================

                    if self.trainedTimesteps % testPer == 0: 

                        averagedRewards = self.test(testSize=testSize)   
                        rewards.append(averagedRewards)

                        #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
                        # TENSORBOARD

                        self.writeTensorboard(rewards[-1], self.trainedEpisodes)

                        #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

                        self.printResult(self.trainedEpisodes, self.trainedTimesteps, rewards[-1])

                    if done or timesteps == self.maxTimesteps-1:
                        break
                # train
                self.__update_weight(Transitions)


        except KeyboardInterrupt:
            print("KeyboardInterruption occured")

            plt.plot(range(len(rewards)), rewards)
        finally:
            plt.plot(range(len(rewards)), rewards)

        self.trainEnv.close()
