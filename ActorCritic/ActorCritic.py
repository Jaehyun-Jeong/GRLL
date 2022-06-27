
import numpy as np
import random
import matplotlib.pyplot as plt

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torch.optim as optim
from torch.autograd import Variable

class ActorCritic():

    '''
    params_dict = {
        'device': device to use, 'cuda' or 'cpu'
        'env': environment like gym
        'model': torch models for policy and value funciton
        'optimizer': torch optimizer
        #MAX_EPISODES = maximum episodes you want to learn
        'maxTimesteps': maximum timesteps agent take 
        'stepsize': GAMMA # step-size for updating Q value
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
        self.stepsize = params_dict['stepsize']

        
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
    def get_action(self, s):
        s = torch.tensor(s).to(self.device)
        _, probs = self.model.forward(s)
        probs = torch.squeeze(probs, 0)
        
        a = probs.multinomial(num_samples=1)
        a = a.data
        
        action = a[0]
        return action
    
    # Returns the action by using epsilon greedy policy in Reinforcment learning
    def epsilon_greedy_action(self, s, epsilon = 0.1):
        s = torch.tensor(s).to(self.device)
        s = torch.unsqueeze(s, 0)
        _, probs = self.model.forward(s)
        
        probs = torch.squeeze(probs, 0)
        
        if random.random() > epsilon:
            a = torch.tensor([torch.argmax(probs)])
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
    def update_weight(self, states, actions, rewards, last_state, entropy_term = 0):

        # compute Q values
        Qval = self.value(last_state)

        # update by using mini-batch Gradient Ascent
        for s_t, a_t, r_tt in reversed(list(zip(states, actions, rewards))):

            log_prob = torch.log(self.pi(s_t, a_t) + self.ups)
            value = self.value(s_t)
            Qval = r_tt + self.stepsize * Qval
            advantage = Variable(Qval - value)

            # get loss
            actor_loss = -(log_prob * advantage)
            critic_loss = -0.5 * (value * advantage).pow(2)
            loss = actor_loss + critic_loss + 0.001 * entropy_term

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def train(self, maxEpisodes, useTensorboard=False):
        try:
            returns = []

            #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
            # TENSORBOARD
            
            if useTensorboard:
                from torch.utils.tensorboard import SummaryWriter
                writer = SummaryWriter()

            #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

            for i_episode in range(maxEpisodes):

                state = self.env.reset()
                init_state = state

                done = False

                states = []
                actions = []
                rewards = [] # no reward at t = 0

                # while not done:
                for timesteps in range(self.maxTimesteps):

                    states.append(state)

                    action = self.get_action(state)
                    actions.append(action)

                    state, reward, done, _ = self.env.step(action.tolist())
                    rewards.append(reward)

                    if done or timesteps == self.maxTimesteps-1:
                        last_state = state
                        break

                self.update_weight(states, actions, rewards, last_state)

                returns.append(sum(rewards))

                #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
                # TENSORBOARD

                if useTensorboard:
                    writer.add_scalars("Returns", {'ActorCritic': returns[-1]}, i_episode)

                #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

                if (i_episode + 1) % 500 == 0:
                    print("Episode: {0:<10} return: {1:<10}".format(i_episode + 1, returns[-1]))

                    # SAVE THE MODEL
                    #torch.save(model, '../saved_models/model' + str(i_episode + 1) + '.pt')

                elif (i_episode + 1) % 10 == 0:
                    print("Episode: {0:<10} return: {1:<10}".format(i_episode + 1, returns[-1]))

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

    MAX_EPISODES = 3000
    MAX_TIMESTEPS = 1000

    ALPHA = 3e-4 # learning rate
    GAMMA = 0.99 # step-size

    # device to use
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # set environment
    env = gym.make("LunarLander-v2")

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
        'stepsize': GAMMA # step-size for updating Q value
    }

    # Initialize Actor-Critic Mehtod
    AC = ActorCritic(**ActorCritic_parameters)

    # TRAIN Agent
    AC.train(MAX_EPISODES, useTensorboard=True)
