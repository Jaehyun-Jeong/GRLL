from typing import Dict, Union

from collections import namedtuple, deque
import numpy as np

# PyTorch
import torch
from torch.autograd import Variable

# Parent Class
from grll.PG.PolicyGradient import PolicyGradient

Transition = namedtuple('Transition',
                        ('state', 'action', 'done', 'next_state', 'reward'))


class A2C(PolicyGradient):

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
        verbose: The verbosity level:
            0 no output,
            1 only train info,
            2 train info + initialized info
        nSteps: Number of Steps for bootstrapping
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
        nSteps: int = 10,
    ):

        # init parameters
        super().__init__(
            trainEnv=trainEnv,
            testEnv=testEnv,
            env=env,
            model=model,
            optimizer=optimizer,
            device=device,
            actionParams=actionParams,
            maxTimesteps=maxTimesteps,
            discount=discount,
            isRender=isRender,
            useTensorboard=useTensorboard,
            tensorboardParams=tensorboardParams,
            clippingParams=clippingParams,
            verbose=verbose,
        )

        # init paramters
        self.nSteps = nSteps
        self.transitions = deque([], maxlen=self.nSteps)

        self.printInit()

    # Update weights by using Actor Critic Method
    def __update_weight(self, entropy_term: float = 0):

        lenLoss = len(self.transitions)
        S_t = [trans.state for trans in self.transitions]
        A_t = [trans.action for trans in self.transitions]
        done = [trans.done for trans in self.transitions]
        S_tt = [trans.next_state for trans in self.transitions]
        R_tt = [trans.reward for trans in self.transitions]

        # Because change list of np.ndarray to tensor is extremly slow
        S_t = np.array(S_t)
        S_t = torch.Tensor(S_t).to(self.device)
        S_tt = np.array(S_tt)
        S_tt = torch.Tensor(S_tt).to(self.device)

        A_t = torch.tensor(A_t).to(self.device)
        done = np.array(done)
        notDone = torch.Tensor(~done).to(self.device)
        R_tt = np.array(R_tt)

        # Compute n-step return
        stateValue = self.value.state_value(S_tt[-1]).unsqueeze(0)
        values = [stateValue * notDone[-1]]
        for r_tt in reversed(R_tt[:-1]):
            values.append(r_tt + self.discount * values[-1])
        values.reverse()
        values = torch.cat(values, 0)

        # get actor loss
        log_prob = self.value.log_prob(S_t, A_t)
        advantage = values - self.value.state_value(S_t)
        advantage = Variable(advantage)  # no grad
        actor_loss = -(advantage * log_prob)

        # get critic loss
        critic_loss = values - self.value.state_value(S_t)
        critic_loss = 1/2 * (critic_loss).pow(2)

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

            while trainTimesteps >= self.trainedTimesteps:

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

                    # trans means transition
                    trans = Transition(state, action, done, next_state, reward)
                    self.transitions.append(trans)

                    state = next_state

                    # Train
                    if len(self.transitions) == self.nSteps or done:
                        self.__update_weight()
                        self.transitions.clear()

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
                        self.transitions.clear()
                        break

        except KeyboardInterrupt:
            print("KeyboardInterruption occured")

        self.trainEnv.close()
