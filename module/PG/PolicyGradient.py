# PyTorch
import torch.nn as nn

from module.RL import RL
from module.utils.ActionSpace import ActionSpace
from module.PG.Value import Value


class PolicyGradient(RL):

    '''
    parameters
        model: torch.nn.Module based model for state_value, and action_value
        optimizer: torch optimizer
        trainEnv: Environment which is used to train
        testEnv: Environment which is used to test
        env: only for when it don't need to be split by trainEnv, testEnv
        device: Device used for training, like Backpropagation
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
        if self.trainEnv.action_space \
                != self.testEnv.action_space:
            raise ValueError(
                    "Action Spaces of trainEnv and testEnv don't match")
        actionSpace = ActionSpace(
                actionSpace=self.trainEnv.action_space)

        self.value = Value(
                model=model.to(self.device),
                device=device,
                optimizer=optimizer,
                actionSpace=actionSpace,
                clippingParams=clippingParams,
                )

        # Stochastic action selection
        self.softmax = nn.Softmax(dim=0)

        # torch.log makes nan(not a number) error,
        # so we have to add some small number in log function
        self.ups = 1e-7
