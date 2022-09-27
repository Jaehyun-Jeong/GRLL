from typing import Dict, Union

import numpy as np
import random
from collections import namedtuple, deque

# PyTorch
import torch
from torch.autograd import Variable

# Parent Class
from grll.PG.PolicyGradient import PolicyGradient

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

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
        exploringParams:
            Exploring parameters selected depanding exploring algorithm
            e.g.)
                When using epsilon greedy
                'exploringParams': {
                    'schedule': 'exponential',
                    'start': 0.99,
                    'end': 0.0001,
                    'decay': 10000
                }
        maxTimesteps: Permitted timesteps in the environment
        discount: Discount rate for calculating return(accumulated reward)
        isRender={

            'train':
            If it's True,
            then render environment screen while training, or vice versa

            'test':
            If it's True,
            then render environment screen while testing, or vice versa

        }
        useTensorboard: False means not using TensorBaord
        tensorboardParams={ TensorBoard parameters
            'logdir': Saved directory
            'tag':
        }
        clippingParams={
            'maxNorm': max value of gradients
            'pNormValue': p value for p-norm
        }
        useBaseline: True means use baseline term for REINFORCE algorithm

        verbose: The verbosity level:
            0 no output,
            1 only train info,
            2 train info + initialized info
    '''

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer,
        trainEnv=None,
        testEnv=None,
        env=None,
        device: torch.device = torch.device('cpu'),
        actionParams: Dict[str, Union[int, float, Dict]] = None,
        maxTimesteps: int = 1000,
        discount: float = 0.99,
        isRender: Dict[str, bool] = {
            'train': False,
            'test': False,
        },
        useTensorboard: bool = False,
        tensorboardParams: Dict[str, str] = {
            'logdir': "./runs/REINFORCE",
            'tag': "Returns"
        },
        clippingParams: Dict[str, Union[int, float]] = {
            'pNormValue': 2,
            'maxNorm': 1,
        },
        verbose: int = 1,
        useBaseline: bool = True,
    ):

        # init parameters
        super().__init__(
            trainEnv=trainEnv,
            testEnv=testEnv,
            env=env,
            model=model,
            actionParams=actionParams,
            optimizer=optimizer,
            device=device,
            maxTimesteps=maxTimesteps,
            discount=discount,
            isRender=isRender,
            useTensorboard=useTensorboard,
            tensorboardParams=tensorboardParams,
            clippingParams=clippingParams,
            verbose=verbose,
        )

        self.useBaseline = useBaseline

        self.printInit()

    # Update weights by using Actor Critic Method
    def __update_weight(self,
                        Transitions: ReplayMemory,
                        entropy_term: float = 0):

        lenLoss = Transitions.memory.__len__()

        S_t = [transition.state for transition in Transitions.memory]
        A_t = [transition.action for transition in Transitions.memory]
        R_tt = [transition.reward for transition in Transitions.memory]

        S_t = np.array(S_t)
        S_t = torch.Tensor(S_t).to(self.device)
        A_t = np.array(A_t)
        A_t = torch.tensor(A_t).to(self.device)

        # calculate Qval
        Qval = torch.Tensor([R_tt[-1]])

        for r_tt in R_tt[:-1]:
            newQval = torch.Tensor([r_tt + self.discount * Qval[0]])
            Qval = torch.cat([newQval, Qval])

        Qval = Qval.to(self.device)

        # get actor loss
        log_prob = self.value.log_prob(S_t, A_t)
        advantage = Qval - self.value.state_value(S_t) * self.useBaseline
        advantage = Variable(advantage)
        actor_loss = -(advantage * log_prob)

        # get critic loss
        value = self.value.state_value(S_t)
        critic_loss = 1/2 * (Qval - value).pow(2)

        # calculate loss
        loss = actor_loss + critic_loss + 0.001 * entropy_term
        loss = torch.sum(loss)/lenLoss

        self.value.step(loss)

    def train(
        self,
        trainTimesteps: int,  # Training Timesteps
        testPer: int = 1000,  # Test per testPer timesteps
        testSize: int = 10,  # The episode size to test
    ):

        try:
            rewards = []

            while trainTimesteps > self.trainedTimesteps:

                Transitions = ReplayMemory(self.maxTimesteps)
                state = self.trainEnv.reset()
                done = False
                self.trainedEpisodes += 1

                # ==========================================================================
                # MAKE TRAIN DATA
                # ==========================================================================

                # while not done:
                for timesteps in range(self.maxTimesteps):
                    self.trainedTimesteps += 1

                    if self.isRender['train']:
                        self.trainEnv.render()

                    action = self.value.get_action(state)

                    next_state, reward, done, _ = self.trainEnv.step(action)
                    Transitions.push(state, action, next_state, reward)
                    state = next_state

                    # ==========================================================================
                    # TEST
                    # ==========================================================================

                    if self.trainedTimesteps % testPer == 0:

                        averagRewards = self.test(testSize=testSize)
                        rewards.append(averagRewards)

                        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
                        # TENSORBOARD

                        self.writeTensorboard(
                                rewards[-1],
                                self.trainedTimesteps)

                        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

                        self.printResult(
                                self.trainedEpisodes,
                                self.trainedTimesteps,
                                rewards[-1])

                    if done or timesteps == self.maxTimesteps-1:
                        break
                # train
                self.__update_weight(Transitions)

        except KeyboardInterrupt:
            print("KeyboardInterruption occured")

        self.trainEnv.close()
