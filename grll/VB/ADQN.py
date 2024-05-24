from typing import Dict, Union

import numpy as np
import random
from collections import namedtuple, deque
from copy import deepcopy

# PyTorch
import torch
import torch.nn.functional as F

from grll.VB.ValueBased import ValueBased
from grll.VB.Value.Value import AveragedValue
from grll.utils.utils import get_action_space

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


class ADQN(ValueBased):

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
        clippingParams={
            'maxNorm': max value of gradients
            'pNormValue': p value for p-norm
        }
        useCheckpoint: False means not using checkpoint
        checkpointParams={
            'metric': metric to save best.  # reward, episode
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
        numPrevModels:
            ADQN averages last k models,
            this parameter determines how many models it save
        updateTargetPer:
            It tells how often target model updates
            If it is 10000, then update target model after 10000 training steps
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
        useCheckpoint: bool = False,
        checkpointParams: Dict[str, str] = {
            'savePath': "./savedModel.obj",
            'metric': 'reward',
        },
        verbose: int = 1,
        gradientStepPer: int = 4,
        epoch: int = 1,
        trainStarts: int = 50000,
        numPrevModels: int = 10,
    ):

        # Init Value Function, Policy
        # Set ActionSpace
        if env:
            actionSpace = get_action_space(
                    env,
                    env)
        elif trainEnv and testEnv:
            actionSpace = get_action_space(
                    trainEnv,
                    testEnv)

        # Only Discrete ActionSpace is possible
        if actionSpace.actionType not in ['Discrete']:
            raise ValueError(
                    "Only support Discrete ActionSpace for DQN!")

        value = AveragedValue(
                model=model,
                device=device,
                optimizer=optimizer,
                actionSpace=actionSpace,
                actionParams=actionParams,
                clippingParams=clippingParams,
                numPrevModels=numPrevModels,
                )

        # init parameters
        super().__init__(
            trainEnv=trainEnv,
            testEnv=testEnv,
            env=env,
            device=device,
            value=value,
            maxTimesteps=maxTimesteps,
            maxMemory=maxMemory,
            discount=discount,
            numBatch=numBatch,
            isRender=isRender,
            useTensorboard=useTensorboard,
            tensorboardParams=tensorboardParams,
            useCheckpoint=useCheckpoint,
            checkpointParams=checkpointParams,
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
            self.value.model.train()
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
                S_tt = np.array(S_tt)
                R_tt = torch.Tensor(np.array(R_tt)).to(self.device)

                actionValue = self.value.pi(S_t, A_t)
                targetValue = self.value.target(
                        S_tt,
                        R_tt,
                        done,
                        self.discount)

                loss = F.mse_loss(targetValue, actionValue)
                loss = loss/lenLoss

                self.value.step(loss)
            self.value.model.eval()

    def train(
        self,
        trainTimesteps: int,  # Training Timesteps
        testPer: int = 1000,  # Test per testPer timesteps
        testSize: int = 10,  # The episode size to test
    ):
        try:
            rewards = []
            # save initial model
            self.value.prevModels.append(deepcopy(self.value.model))

            spentTimesteps = 0  # spent timesteps after starting train
            while trainTimesteps > spentTimesteps:

                # Second parameter is information
                state, _ = self.trainEnv.reset()
                done = False
                self.trainedEpisodes += 1

                # MAKE TRAIN DATA
                # while not done:
                for timesteps in range(self.maxTimesteps):
                    spentTimesteps += 1
                    self.trainedTimesteps += 1

                    if self.isRender['train']:
                        self.trainEnv.render()

                    action = self.value.get_action(state)

                    next_state, reward, terminal, truncated, _ \
                        = self.trainEnv.step(action)
                    done = terminal or truncated

                    self.replayMemory.push(
                            state,
                            action,
                            done,
                            next_state,
                            reward)

                    state = next_state

                    # train
                    self.update_weight()
                    # save updated model
                    self.value.prevModels.append(deepcopy(self.value.model))

                    # TEST
                    if spentTimesteps % testPer == 0:

                        meanReward, meanEpisode = self.test(testSize=testSize)
                        rewards.append(meanReward)

                        # TENSORBOARD
                        self.writeTensorboard(
                                rewards[-1],
                                self.trainedTimesteps)

                        self.printResult(
                                self.trainedEpisodes,
                                self.trainedTimesteps,
                                rewards[-1],
                                meanEpisode)

                    if done \
                            or timesteps == self.maxTimesteps-1 \
                            or spentTimesteps >= trainTimesteps:
                        break

        except KeyboardInterrupt:
            print("KeyboardInterruption occured")

        self.trainEnv.close()
