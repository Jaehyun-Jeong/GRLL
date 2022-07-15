
import numpy as np
import random 
import matplotlib.pyplot as plt
from collections import namedtuple, deque

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torch.optim as optim
from torch.autograd import Variable

# Parent Class
from module.ActorCritic.ActorCritic import ActorCritic

Transition = namedtuple('Transition',
                       ('state', 'action', 'done', 'next_state', 'reward'))

class onestep_ActorCritic(ActorCritic):

    '''
    params_dict = {
        'device': device to use, 'cuda' or 'cpu'
        'env': environment like gym
        'model': torch models for policy and value funciton
        'optimizer': torch optimizer
        #MAX_EPISODES = maximum episodes you want to learn
        'maxTimesteps': maximum timesteps agent take 
        'discount_rate': GAMMA # step-size for updating Q value
        'eps': {
            'start': 0.9,
            'end': 0.05,
            'decay': 200
        }, 
        'trainPolicy': select from greedy, eps-greedy, stochastic, eps-stochastic
    }
    '''

    def __init__(
        self, 
        env,
        model,
        optimizer,
        device=torch.device('cpu'),
        eps={
            'start': 0.99,
            'end': 0.0001,
            'decay': 10000
        },
        maxTimesteps=1000,
        discount_rate=0.99,
        isRender={
            'train': False,
            'test': False,
        },
        useTensorboard=False,
        tensorboardParams={
            'logdir': "./runs/onestep_ActorCritic",
            'tag': "Returns"
        },
        policy={
            'train': 'eps-stochastic',
            'test': 'stochastic'
        },
    ):

        # init parameters 
        super().__init__(
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
            policy=policy
        )

    # Update weights by using Actor Critic Method
    def update_weight(self, Transition, entropy_term = 0):
        s_t = Transition.state
        a_t = Transition.action
        s_tt = Transition.next_state
        r_tt = Transition.reward
        done = Transition.done

        # get actor loss
        log_prob = torch.log(self.pi(s_t, a_t) + self.ups)
        advantage = Variable(r_tt + self.value(s_tt)*(not done) - self.value(s_t))
        actor_loss = -(advantage * log_prob)

        # get critic loss
        critic_loss = 1/2 * (r_tt + self.value(s_tt)*(not done) - self.value(s_t)).pow(2)

        loss = actor_loss + critic_loss + 0.001 * entropy_term
    
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.steps_done += 1

    def train(
        self, 
        maxEpisodes, 
        testPer=10, 
        testSize=10,
    ):

        try:
            returns = []
            
            for i_episode in range(maxEpisodes):
                
                state = self.env.reset()
                done = False
                
                #==========================================================================
                # MAKE TRAIN DATA
                #==========================================================================

                # while not done:
                for timesteps in range(self.maxTimesteps):

                    if self.isRender['train']:
                        env.render()

                    action = self.get_action(state, useEps=self.useTrainEps, useStochastic=self.useTrainStochastic)
                    next_state, reward, done, _ = self.env.step(action.tolist())

                    # trans means transition 
                    trans = Transition(state, action, done, next_state, reward)

                    state = next_state

                    # Train
                    if done or timesteps == self.maxTimesteps-1:
                        break
                    
                    self.update_weight(trans)

                #==========================================================================
                # TEST
                #==========================================================================

                if (i_episode+1) % testPer == 0: 

                    cumulative_rewards = self.test(testSize=testSize)   
                    returns.append(cumulative_rewards)

                    #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
                    # TENSORBOARD

                    self.writeTensorboard(returns[-1], i_episode+1)

                    #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

                    self.printResult(i_episode+1, returns[-1])

        except KeyboardInterrupt:
            print("==============================================")
            print("KEYBOARD INTERRUPTION!!=======================")
            print("==============================================")

            plt.plot(range(len(returns)), returns)
        finally:
            plt.plot(range(len(returns)), returns)

        self.env.close()

if __name__ == "__main__":

    from models import ANN_V1 # import model
    import gym # Environment 

    MAX_EPISODES = 10000
    MAX_TIMESTEPS = 1000

    ALPHA = 0.1e-3 # learning rate
    GAMMA = 0.99 # discount_rate
    epsilon = 0 # for epsilon greedy action

    # device to use
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # set environment
    env = gym.make("CartPole-v0")
    #env = gym.make("Acrobot-v1")
    #env = gym.make("MountainCar-v0")

    num_actions = env.action_space.n
    num_states = env.observation_space.shape[0]

    ACmodel = ANN_V1(num_states, num_actions).to(device)
    optimizer = optim.Adam(ACmodel.parameters(), lr=ALPHA)

    ActorCritic_parameters = {
        'device': device, # device to use, 'cuda' or 'cpu'
        'env': env, # environment like gym
        'model': ACmodel, # torch models for policy and value funciton
        'optimizer': optimizer, # torch optimizer
        'maxTimesteps': MAX_TIMESTEPS, # maximum timesteps agent take 
        'discount_rate': GAMMA, # step-size for updating Q value
        'epsilon': epsilon, # epsilon greedy action for training
    }

    # Initialize ActorCritic Mehtod
    AC = ActorCritic(**ActorCritic_parameters)

    # TRAIN Agent
    AC.train(MAX_EPISODES, isRender=False, useTensorboard=True, tensorboardTag="CartPole-v1")
