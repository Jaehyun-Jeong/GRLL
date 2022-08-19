from typing import Dict, Union

from collections import namedtuple
import numpy as np

# PyTorch
import torch
from torch.autograd import Variable

# Parent Class
from module.PolicyGradient.PolicyGradient import PolicyGradient

Transition = namedtuple('Transition',
                        ('state', 'action', 'done', 'next_state', 'reward'))


class onestep_ActorCritic(PolicyGradient):

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
        model: torch.nn.Module,
        optimizer,
        trainEnv=None,
        testEnv=None,
        env=None,
        device: torch.device = torch.device('cpu'),
        eps: Dict[str, Union[int, float]] = {
            'start': 0.99,
            'end': 0.0001,
            'decay': 10000
        },
        maxTimesteps: int = 1000,
        discount_rate: float = 0.99,
        isRender: Dict[str, bool] = {
            'train': False,
            'test': False,
        },
        useTensorboard: bool = False,
        tensorboardParams: Dict[str, str] = {
            'logdir': "./runs/REINFORCE",
            'tag': "Returns"
        },
        policy: Dict[str, str] = {
            'train': 'eps-stochastic',
            'test': 'stochastic'
        },
        verbose: int = 1,
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

        self.printInit()

    # Overrided method from PolicyGradient for single pi
    def pi(
            self,
            s: Union[torch.Tensor, np.ndarray],
            a: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:

        s = torch.Tensor(s).to(self.device)
        _, probs = self.model.forward(s)

        return probs[a]

    # Overrided method from PolicyGradient for single state value
    def value(self, s: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        s = torch.Tensor(s).to(self.device)
        value, _ = self.model.forward(s)
        value = torch.squeeze(value, 0)

        return value

    # Update weights by using Actor Critic Method
    def update_weight(self, Transition: Transition, entropy_term: float = 0):
        s_t = Transition.state
        a_t = Transition.action
        s_tt = Transition.next_state
        r_tt = Transition.reward
        done = Transition.done

        # get actor loss
        log_prob = torch.log(self.pi(s_t, a_t) + self.ups)
        advantage = r_tt + self.value(s_tt)*(not done) - self.value(s_t)
        advantage = Variable(advantage)  # no grad
        actor_loss = -(advantage * log_prob)

        # get critic loss
        critic_loss = r_tt + self.value(s_tt)*(not done) - self.value(s_t)
        critic_loss = 1/2 * (critic_loss).pow(2)

        loss = actor_loss + critic_loss + 0.001 * entropy_term

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.steps_done += 1

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

                    action = self.get_action(
                            state,
                            useEps=self.useTrainEps,
                            useStochastic=self.useTrainStochastic)

                    next_state, reward, done, _ = self.trainEnv.step(action)

                    # trans means transition
                    trans = Transition(state, action, done, next_state, reward)

                    state = next_state

                    # Train
                    self.update_weight(trans)

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
                                self.trainedEpisodes)

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
