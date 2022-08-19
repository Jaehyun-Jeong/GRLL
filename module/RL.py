from typing import Union

from datetime import datetime, timedelta
import torch
import numpy as np


class RL():

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
        maxTimesteps,
        eps,
        policy,
        device,
        isRender,
        useTensorboard,
        tensorboardParams,
        verbose,
    ):

        # set Environment
        if env is None and trainEnv is not None and testEnv is not None:
            self.trainEnv = trainEnv
            self.testEnv = testEnv
        elif env is not None and trainEnv is None and testEnv is None:
            from copy import deepcopy

            self.trainEnv = env
            self.testEnv = deepcopy(env)
        else:
            raise ValueError(
                "No Environment or you just use one of trainEnv, or testEnv,"
                "or you set the env with trainEnv or testEnv,"
                "or you set the trainEnv or testEnv with env"
            )

        # init parameters
        self.device = device
        self.model = model.to(self.device)  # set dtype to match
        self.optimizer = optimizer
        self.maxTimesteps = maxTimesteps
        self.policy = policy
        self.isRender = isRender
        self.useTensorboard = useTensorboard
        self.tensorboardParams = tensorboardParams
        self.verbose = verbose

        # Init trained Episode
        self.trainedEpisodes = 0
        self.trainedTimesteps = 0

        # check the time spent
        self.timePrevStep = datetime.now()
        self.timeSpent = timedelta(0)

        # init Summary Writer
        if self.useTensorboard:
            try:
                from tensorboardX import SummaryWriter
                self.tensorboardWriter = \
                    SummaryWriter(self.tensorboardParams['logdir'])
            except ImportError:
                ImportError("tensorboardX does not exist")

        # ==================================================================================
        # select train, test policy
        # ==================================================================================

        policyDict = {
                # [ useEpsilon, useStochastic ]
                'greedy': [False, False],
                'stochastic': [False, True],
                'eps-greedy': [True, False],
                'eps-stochastic': [True, True]}

        if not self.policy['train'] in policyDict.keys() or \
                not self.policy['test'] in policyDict.keys():
            raise ValueError("Possible policies are \
                    'greedy', 'eps-greedy', \
                    'stochastic', and 'eps-stochastic'")

        trainPolicyList = policyDict[self.policy['train']]
        testPolicyList = policyDict[self.policy['test']]

        if trainPolicyList[0] or testPolicyList[0]:
            self.eps = eps

        self.useTrainEps = trainPolicyList[0]
        self.useTrainStochastic = trainPolicyList[1]
        self.useTestEps = testPolicyList[0]
        self.useTestStochastic = testPolicyList[1]

        # ==================================================================================

    # Draw graph in TensorBoard only when It use TensorBoard
    def writeTensorboard(self, y: Union[float, str], x: int):
        if self.useTensorboard:
            self.tensorboardWriter.add_scalar(
                    self.tensorboardParams['tag'], y, x)

    # get_action method to be overrided
    def get_action(
            self,
            s: Union[torch.Tensor, np.ndarray],
            useEps: bool,
            useStochastic: bool) -> torch.Tensor:

        pass

    # Test to measure performance
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

                action = self.get_action(
                        state,
                        useEps=self.useTestEps,
                        useStochastic=self.useTestStochastic)

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

    # Print all Initialized Properties
    def printInit(self):
        if self.verbose >= 2:  # Print After checking verbosity level
            try:
                # Not working with nohup command
                import os
                printLength = os.get_terminal_size().columns
            except OSError:
                printLength = 30

            print("="*printLength+"\n")
            print("Initialized Parameters\n")

            for key, value in self.__dict__.items():
                print(f"{key}: {value}")

            print("\n"+"="*printLength)

    # Print all measured performance
    def printResult(
            self,
            episode: int,
            timesteps: int,
            averageReward: Union[str, float]):

        if self.verbose >= 1:  # Print After checking verbosity level
            try:
                # Not working with nohup command
                import os
                printLength = os.get_terminal_size().columns
            except OSError:
                printLength = 30000

            self.timeSpent += datetime.now() - self.timePrevStep

            results = \
                f"| Timesteps / Episode : {str(timesteps)[0:10]:>10} "\
                f"/ {str(episode)[0:10]:>10} "\
                f"| Averag Reward: {str(averageReward)[0:10]:>10} "\
                f"| Time Spent : {str(self.timeSpent):10} "\
                f"/ {str(datetime.now()-self.timePrevStep):10} | "

            splited = results.split('|')[1:-1]
            frameString = "+"

            for split in splited:
                frameString += "-"*len(split) + "+"

            if len(frameString) > printLength:
                frameString = frameString[:printLength]
            if len(results) > printLength:
                results = results[:printLength-3]
                results += "..."

            print(frameString)
            print(results)
            print(frameString)

            self.timePrevStep = datetime.now()

    # save class
    def save(self, saveDir: str = str(datetime)+".obj"):

        save_dict = self.__dict__

        # belows are impossible to dump
        save_dict.pop('tensorboardWriter', None)
        save_dict.pop('trainEnv', None)
        save_dict.pop('testEnv', None)
        save_dict.pop('device', None)

        # save model state dict
        save_dict['modelStateDict'] = save_dict['model'].state_dict()
        save_dict.pop('model', None)
        save_dict['optimizerStateDict'] = save_dict['optimizer'].state_dict()
        save_dict.pop('optimizer', None)

        torch.save(save_dict, saveDir)

    # load class
    def load(self, loadDir: str):

        # =============================================================================
        # LOAD TORCH MODEL
        # =============================================================================

        loadedDict = torch.load(loadDir, map_location=self.device)

        try:
            self.model.load_state_dict(
                    loadedDict.pop('modelStateDict'))
            self.optimizer.load_state_dict(
                    loadedDict.pop('optimizerStateDict'))
        except ValueError:
            raise ValueError("No matching torch.nn.Module, \
                    please use equally shaped torch.nn.Module as you've done!")

        self.model.eval()

        # =============================================================================

        for key, value in loadedDict.items():
            self.__dict__[key] = value

        self.timePrevStep = datetime.now()  # Recalculating time spent
