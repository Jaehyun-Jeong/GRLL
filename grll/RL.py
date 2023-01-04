from typing import Union
import warnings

from datetime import datetime, timedelta
import torch


class RL():

    '''
    parameters
        model: torch.nn.Module based model for state_value, and action_value
        optimizer: torch optimizer
        trainEnv: Environment which is used to train
        testEnv: Environment which is used to test
        env: only for when it don't need to be split by trainEnv, testEnv
        device: Device used for training, like Backpropagation
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
        maxTimesteps,
        discount,
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

        # Init parameters
        self.device = device
        self.maxTimesteps = maxTimesteps
        self.discount = discount
        self._isRender = isRender
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

    @property
    def isRender(self):
        return self._isRender

    @isRender.setter
    def isRender(self, isRender):
        self._isRender = isRender

    # Draw graph in TensorBoard only when It use TensorBoard
    def writeTensorboard(self, y: Union[float, str], x: int):
        if self.useTensorboard:
            self.tensorboardWriter.add_scalar(
                    self.tensorboardParams['tag'], y, x)

    # Test to measure performance
    def test(
            self,
            testSize: int) -> Union[float, str]:

        raise NotImplementedError()

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
            meanReward: Union[str, float],
            meanEpisode: Union[str, float]):

        if self.verbose >= 1:  # Print After checking verbosity level
            try:
                # Not working with nohup command
                import os
                printLength = os.get_terminal_size().columns
            except OSError:
                printLength = 30000

            self.timeSpent += datetime.now() - self.timePrevStep

            Result = [
                f"| Timesteps / Episode :"
                f" {str(timesteps)[0:10]:>10} / {str(episode)[0:10]:>10} |",
                f"| Reward Mean         :"
                f" {str(meanReward)[0:10]:>20}    |",
                f"| Episode Mean        :"
                f" {str(meanEpisode)[0:10]:>20}    |",
                f"| Time Spent (step)   :"
                f" {str(datetime.now()-self.timePrevStep):>20}    |",
                f"| Time Spent (train)  :"
                f" {str(self.timeSpent):>20}    |"]

            frameLen = len(Result[0])-2  # -2 exist to make + shape frames
            frameString = "+"
            frameString += "-"*frameLen + "+"

            if len(frameString) > printLength:
                frameString = frameString[:printLength]
            if len(Result[0]) > printLength:
                Result[0] = Result[0][:printLength-3]
                Result[1] = Result[1][:printLength-3]
                Result[2] = Result[2][:printLength-3]
                Result[3] = Result[3][:printLength-3]
                Result[4] = Result[4][:printLength-3]
                Result[0] += "..."
                Result[1] += "..."
                Result[2] += "..."
                Result[3] += "..."
                Result[4] += "..."

            results = ""
            results += Result[0] + "\n"
            results += Result[1] + "\n"
            results += Result[2] + "\n"
            results += Result[3] + "\n"
            results += Result[4]

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
        save_dict['modelStateDict'] \
            = save_dict['value'].model.state_dict()
        save_dict['value'].model = None
        save_dict['optimizerStateDict'] \
            = save_dict['value'].optimizer.state_dict()
        save_dict['value'].optimizer = None

        torch.save(save_dict, saveDir)

    # Load class
    def load(self, loadDir: str):

        # Load torch model
        loadedDict = torch.load(loadDir, map_location=self.device)

        # Load state_dict of torch model, and optimizer
        try:

            self.value.model.load_state_dict(
                    loadedDict.pop('modelStateDict'))
            self.value.optimizer.load_state_dict(
                    loadedDict.pop('optimizerStateDict'))

            loadedDict['value'].__dict__.pop('model')
            loadedDict['value'].__dict__.pop('optimizer')

            # Load value function
            for key, value in loadedDict['value'].__dict__.items():
                self.value.__dict__[key] = value
            loadedDict.pop('value')

        except ValueError:
            print(
                "No matching torch.nn.Module,"
                "please use equally shaped torch.nn.Module as you've done!")

        self.value.model.eval()

        if self.device:
            loadedDict.pop('device')
        else:
            warnings.warn(
                    "Please use device parameter to check your device\n"
                    "It could be a problem, if you were using CUDA, but now different device")

        for key, value in loadedDict.items():
            self.__dict__[key] = value

        self.timePrevStep = datetime.now()  # Recalculating time spent
