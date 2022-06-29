
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


Transition = namedtuple('Transition',
                       ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class ActorCritic():

    '''
    params_dict = {
        'device': device to use, 'cuda' or 'cpu'
        'env': environment like gym
        'model': torch models for policy and value funciton
        'optimizer': torch optimizer
        #MAX_EPISODES = maximum episodes you want to learn
        'maxTimesteps': maximum timesteps agent take 
        'discount_rate': GAMMA # step-size for updating Q value
        'epsilon': epsilon for epsilon greedy action
    }
    '''

    def __init__(self, **params_dict):
        super(ActorCritic, self).__init__()

        # init parameters 
        self.device = params_dict['device']
        self.env = params_dict['env']
        self.model = params_dict['model']
        self.optimizer = params_dict['optimizer']
        self.maxTimesteps = params_dict['maxTimesteps'] 
        self.discount_rate = params_dict['discount_rate']
        self.epsilon = params_dict['epsilon']

        
        # torch.log makes nan(not a number) error, so we have to add some small number in log function
        self.ups=1e-7

    # In Reinforcement learning, pi means the function from state space to action probability distribution
    # Returns probability of taken action a from state s
    def pi(self, s, a):
        s = torch.Tensor(s).to(self.device)
        _, probs = self.model.forward(s)
        probs = torch.squeeze(probs, 0)
        return probs[a]
    
    # Returns the action from state s by using multinomial distribution
    def get_action(self, s, epsilon = 0):
        with torch.no_grad():
            s = torch.tensor(s).to(self.device)
            _, probs = self.model.forward(s)
            probs = torch.squeeze(probs, 0)
            
            if random.random() > epsilon:
                a = probs.multinomial(num_samples=1)
            else:
                a = torch.rand(probs.shape).multinomial(num_samples=1)
            
            a = a.data
            action = a[0]
            return action
  
    # Returns a value of the state (state value function in Reinforcement learning)
    def value(self, s):
        s = torch.tensor(s).to(self.device)
        value, _ = self.model.forward(s)
        value = torch.squeeze(value, 0)

        return value    

    # Update weights by using Actor Critic Method
    def update_weight(self, Transitions, entropy_term = 0):
        # update by using mini-batch Gradient Ascent
        for Transition in reversed(Transitions.memory):
            s_t = Transition.state
            a_t = Transition.action
            s_tt = Transition.next_state
            r_tt = Transition.reward

            # get actor loss
            log_prob = torch.log(self.pi(s_t, a_t) + self.ups)
            advantage = r_tt + self.value(s_tt) - self.value(s_t)
            actor_loss = -(advantage * log_prob)

            # get critic loss
            value = self.value(s_t)
            next_value = self.value(s_tt)
            critic_loss = 1/2 * (r_tt + self.discount_rate * next_value - value).pow(2)

            loss = actor_loss + critic_loss + 0.001 * entropy_term

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def train(self, maxEpisodes, testPer=10, isRender=False, useTensorboard=False, tensorboardTag="ActorCritic"):
        try:
            train_returns = []
            test_returns = []
            
            #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
            # TENSORBOARD
            
            if useTensorboard:
                from torch.utils.tensorboard import SummaryWriter
                writer = SummaryWriter()

            #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

            for i_episode in range(maxEpisodes):
                
                Transitions = ReplayMemory(maxEpisodes)
                state = self.env.reset()
                done = False
                train_rewards = []
                test_rewards = []
                
                #==========================================================================
                # MAKE TRAIN DATA
                #==========================================================================

                # while not done:
                for timesteps in range(self.maxTimesteps):

                    if isRender:
                        env.render()

                    action = self.get_action(state, epsilon=epsilon)

                    print(action)

                    next_state, reward, done, _ = self.env.step(action.tolist())
                    
                    Transitions.push(state, action, next_state, reward)

                    train_rewards.append(reward)
                    state = next_state

                    if done or timesteps == self.maxTimesteps-1:
                        break

                self.update_weight(Transitions)
                train_returns.append(sum(train_rewards))

                #==========================================================================
                # TEST
                #==========================================================================

                if (i_episode+1) % testPer == 0: 
                    self.env.reset()

                    for timesteps in range(self.maxTimesteps):
                        if isRender:
                            env.render()

                        action = self.get_action(state)
                        _, reward, done, _ = self.env.step(action.tolist())

                        test_rewards.append(reward)

                        if done or timesteps == self.maxTimesteps-1:
                            break

                    test_returns.append(sum(test_rewards))

                    #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
                    # TENSORBOARD

                    if useTensorboard:
                        writer.add_scalars("Returns", {tensorboardTag: test_returns[-1]}, i_episode)

                    #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

                if (i_episode + 1) % 500 == 0:
                    print("Episode: {0:<10} return: {1:<10}".format(i_episode + 1, test_returns[-1]))
                elif (i_episode + 1) % 10 == 0:
                    print("Episode: {0:<10} return: {1:<10}".format(i_episode + 1, test_returns[-1]))

        except KeyboardInterrupt:
            print("==============================================")
            print("KEYBOARD INTERRUPTION!!=======================")
            print("==============================================")

            plt.plot(range(len(test_returns)), test_returns)
        finally:
            plt.plot(range(len(test_returns)), test_returns)

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

    # set ActorCritic
    num_actions = env.action_space.n
    num_states = env.observation_space.shape[0]

    ACmodel = ANN_V1(num_states, num_actions).to(device)
    optimizer = optim.Adam(ACmodel.parameters(), lr=ALPHA)

    ActorCritic_parameters = {
        'device': device, # device to use, 'cuda' or 'cpu'
        'env': env, # environment like gym
        'model': ACmodel, # torch models for policy and value funciton
        'optimizer': optimizer, # torch optimizer
        #MAX_EPISODES = MAX_EPISODES, # maximum episodes you want to learn
        'maxTimesteps': MAX_TIMESTEPS, # maximum timesteps agent take 
        'discount_rate': GAMMA, # step-size for updating Q value
        'epsilon': epsilon # epsilon greedy action for training
    }

    # Initialize Actor-Critic Mehtod
    AC = ActorCritic(**ActorCritic_parameters)

    # TRAIN Agent
    AC.train(MAX_EPISODES, isRender=False, useTensorboard=True, tensorboardTag="CartPole-v1")
