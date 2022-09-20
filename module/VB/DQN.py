from typing import Dict, Union

import numpy as np
import random
from collections import namedtuple, deque

# PyTorch
import torch
from torch.autograd import Variable

from module.VB.ValueBased import ValueBased

Transition = namedtuple('Transition',
                        ('state', 'action', 'done', 'next_state', 'reward'))


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


class DQN(ValueBased):

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
        discount: Discount rate for calculating return(accumulated reward)
        maxMemory: Memory size for Experience Replay
        numBatch: Batch size for mini-batch gradient descent
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
        policy={

            there are 4 types of Policy
            'stochastic',
            'eps-stochastic',
            'greedy',
            'eps-greedy'

            'train': e.g. 'eps-stochastic'
            'test': e.g. 'stochastic'
        }
        clippingParams={
            'maxNorm': max value of gradients
            'pNormValue': p value for p-norm
        }
        verbose: The verbosity level:
            0 no output,
            1 only train info,
            2 train info + initialized info
        gradientStepPer:
            Update the neural network model every gradientStepPer timesteps
        epoch: Epoch size to train from given data(Replay Memory)
        trainStarts:
            how many steps of the model
            to collect transitions for before learning starts
    '''

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer,
        trainEnv=None,
        testEnv=None,
        env=None,
        device: torch.device = torch.device('cpu'),
        maxTimesteps: int = 1000,
        discount: float = 0.99,
        maxMemory: int = 100000,
        numBatch: int = 64,
        isRender: Dict[str, bool] = {
            'train': False,
            'test': False
        },
        useTensorboard: bool = False,
        tensorboardParams: Dict[str, str] = {
            'logdir': "./runs",
            'tag': "DQN"
        },
        clippingParams: Dict[str, Union[int, float]] = {
            'pNormValue': 2,
            'maxNorm': 1,
        },
        verbose: int = 1,
        gradientStepPer: int = 4,
        epoch: int = 1,
        trainStarts: int = 50000,
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
            discount=discount,
            numBatch=numBatch,
            isRender=isRender,
            useTensorboard=useTensorboard,
            tensorboardParams=tensorboardParams,
            clippingParams=clippingParams,
            verbose=verbose,
            gradientStepPer=gradientStepPer,
            epoch=epoch,
            trainStarts=trainStarts,
        )

        self.replayMemory = ReplayMemory(maxMemory)

        self.printInit()

    # Update weights by using Actor Critic Method
    def update_weight(self):
        if self.is_trainable():
            for _ in range(self.epoch):

                # if memory is smaller then numBatch, then just use all data
                batch_size = self.numBatch \
                    if self.numBatch <= self.replayMemory.memory.__len__() \
                    else self.replayMemory.__len__()

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

                actionValue = self.value.pi(S_t, A_t)
                nextMaxValue = self.value.max_value(S_tt)

                target = R_tt + self.discount * nextMaxValue * notDone
                target = Variable(target)  # No grad
                loss = 1/2 * (target - actionValue).pow(2)
                loss = torch.sum(loss)/lenLoss

                self.step(loss)

    def train(
        self,
        trainTimesteps: int,  # Training Timesteps
        testPer: int = 1000,  # Test per testPer timesteps
        testSize: int = 10,  # The episode size to test
    ):

        try:
            rewards = []

            while trainTimesteps > self.trainedTimesteps:

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
                    self.replayMemory.push(
                            state,
                            action,
                            done,
                            next_state,
                            reward)

                    state = next_state

                    # train
                    self.update_weight()

                    # ==========================================================================
                    # TEST
                    # ==========================================================================
                    if (self.trainedTimesteps) % testPer == 0:

                        averageRewards = self.test(testSize=testSize)
                        rewards.append(averageRewards)

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

        except KeyboardInterrupt:
            print("KeyboardInterruption occured")

        self.trainEnv.close()
