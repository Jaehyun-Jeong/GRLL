from typing import Union

# PyTorch
import torch.nn as nn

from module.RL import RL
from module.utils.utils import overrides


class ValueBased(RL):

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
        algorithm: algorithm it use
            e.g.) DQN, ADQN
    '''

    def __init__(
        self,
        trainEnv,
        testEnv,
        env,
        device,
        value,
        maxTimesteps,
        maxMemory,
        discount,
        numBatch,
        isRender,
        useTensorboard,
        tensorboardParams,
        verbose,
        gradientStepPer,
        epoch,
        trainStarts,
    ):

        # init parameters
        super().__init__(
            trainEnv=trainEnv,
            testEnv=testEnv,
            env=env,
            device=device,
            maxTimesteps=maxTimesteps,
            discount=discount,
            isRender=isRender,
            useTensorboard=useTensorboard,
            tensorboardParams=tensorboardParams,
            verbose=verbose
        )

        self.value = value

        # Init
        self.maxMemory = maxMemory
        self.numBatch = numBatch
        self.gradientStepPer = gradientStepPer
        self.epoch = epoch
        self.trainStarts = trainStarts

        # Stochastic action selection
        self.softmax = nn.Softmax(dim=0)

        # torch.log makes nan(not a number) error,
        # so we have to add some small number in log function
        self.ups = 1e-7

    def is_trainable(self):

        # check train condition
        condition = True if self.trainedTimesteps % self.gradientStepPer == 0 \
                    and self.trainedTimesteps >= self.trainStarts \
                    else False

        return condition

    # Test to measure performance
    @overrides(RL)
    def test(
            self,
            testSize: int) -> Union[float, str]:

        rewards = []

        for _ in range(testSize):

            state = self.testEnv.reset()
            done = False
            cumulativeRewards = 0

            for timesteps in range(self.maxTimesteps):
                if self.isRender['test']:
                    self.testEnv.render()

                action = self.value.get_action(state)

                next_state, reward, done, _ = self.testEnv.step(action)

                cumulativeRewards += reward
                state = next_state

                if done or timesteps == self.maxTimesteps-1:
                    break

            rewards.append(cumulativeRewards)

        if testSize > 0:
            return sum(rewards) / testSize  # Averaged Rewards
        elif testSize == 0:
            return "no Test"
        else:
            raise ValueError("testSize can't be smaller than 0")
