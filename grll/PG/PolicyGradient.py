from typing import Union

from grll.RL import RL
from grll.utils.ActionSpace import ActionSpace
from grll.utils.utils import overrides, get_action_space
from grll.PG.Value import Value


class PolicyGradient(RL):

    '''
    parameters
        model: torch.nn.Module based model for state_value, and action_value
        optimizer: torch optimizer
        trainEnv: Environment which is used to train
        testEnv: Environment which is used to test
        env: only for when it don't need to be split by trainEnv, testEnv
        device: Device used for training, like Backpropagation
        actionParameters={
            # for DISCRETE
            'algorithm': "greedy",  # greedy, stochastic
            'exploring': "epsilon",  # epsilon, None
            'exploringParams': {
                'start': 0.99,
                'end': 0.0001,
                'decay': 10000
            }

            # for CONTINUOUS
            'algorithm': "plain",  # greedy
            'exploring': "normal",  # normal
            'exploringParams': {
                'mean': 0,
                'sigma': 1,
            }
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
        verbose: The verbosity level:
            0 no output,
            1 only train info,
            2 train info + initialized info
    '''

    def __init__(
        self,
        trainEnv,
        testEnv,
        env,
        model,
        optimizer,
        device,
        actionParams,
        maxTimesteps,
        discount,
        isRender,
        useTensorboard,
        tensorboardParams,
        clippingParams,
        verbose,
    ):

        # init parameters
        super().__init__(
            device=device,
            trainEnv=trainEnv,
            testEnv=testEnv,
            env=env,
            maxTimesteps=maxTimesteps,
            discount=discount,
            isRender=isRender,
            useTensorboard=useTensorboard,
            tensorboardParams=tensorboardParams,
            verbose=verbose,
        )

        # Init Value Function, Policy
        # Set ActionSpace
        actionSpace = get_action_space(
               self.trainEnv,
               self.testEnv)

        # Support 'Discrete', 'Continuous' ActionSpaces
        if actionSpace.actionType not in ['Discrete', 'Continuous']:
            raise ValueError(
                    "Only support Discrete ActionSpace for DQN!")

        self.value = Value(
                model=model.to(self.device),
                device=device,
                optimizer=optimizer,
                actionSpace=actionSpace,
                actionParams=actionParams,
                clippingParams=clippingParams,
                )

    # Test to measure performance
    @overrides(RL)
    def test(
            self,
            testSize: int) -> Union[float, str]:

        # Calculate mean rewards
        rewards = []
        # Calculate mean episode length
        episodesLen = []

        for _ in range(testSize):

            state = self.testEnv.reset()
            done = False
            cumulativeRewards = 0
            episodeLen = 0

            for timesteps in range(self.maxTimesteps):
                if self.isRender['test']:
                    self.testEnv.render()

                action = self.value.get_action(
                        state,
                        isTest=True)

                next_state, reward, done, _ = self.testEnv.step(action)

                cumulativeRewards += reward
                episodeLen += 1
                state = next_state

                if done or timesteps == self.maxTimesteps-1:
                    break

            episodesLen.append(episodeLen)
            rewards.append(cumulativeRewards)

        if testSize > 0:
            meanEpisode = sum(episodesLen) / testSize
            meanReward = sum(rewards) / testSize  # Averaged Rewards

            return meanReward, meanEpisode
        elif testSize == 0:
            return "no Test", "no Test"
        else:
            raise ValueError("testSize can't be smaller than 0")
