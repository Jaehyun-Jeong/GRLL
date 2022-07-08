
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
        device="cpu", 
        maxTimesteps=1000,
        discount_rate=0.99,
        eps={
            'start': 0.9,
            'end': 0.05,
            'decay': 200
        },
        trainPolicy='eps-stochastic',
        testPolicy='stochastic'
    ):

        super(ActorCritic, self).__init__()

        # init parameters 
        self.device = device
        self.env = env
        self.model = model
        self.optimizer = optimizer
        self.maxTimesteps = maxTimesteps 
        self.discount_rate = discount_rate
        self.steps_done = 0 # for epsilon scheduling

        # select train, test policy
        policyDict = {'greedy': [False, False], 'stochastic': [False, True], 'eps-greedy': [True, False], 'eps-stochastic': [True, True]} # [ useEpsilon, useStochastic ]

        try:
            trainPolicyList = policyDict[trainPolicy]
            testPolicyList = policyDict[testPolicy]

            if trainPolicyList[0] or testPolicyList[0]:
                self.eps = eps

            self.useTrainEps = trainPolicyList[0]
            self.useTrainStochastic = trainPolicyList[1]
            self.useTestEps = testPolicyList[0]
            self.useTestStochastic = testPolicyList[1]

        except: 
            print("ERROR OCCURED : supported policies are 'greedy', 'eps-greedy', 'stochastic', and 'eps-stochastic'")
        
        # torch.log makes nan(not a number) error, so we have to add some small number in log function
        self.ups=1e-7

    def get_eps(self):
        import math

        eps_start = self.eps['start']
        eps_end = self.eps['end']
        eps_decay = self.eps['decay']

        return eps_end + (eps_start + eps_end) * math.exp(-1. * self.steps_done / eps_decay)

    # In Reinforcement learning, pi means the function from state space to action probability distribution
    # Returns probability of taken action a from state s
    def pi(self, s, a):
        s = torch.Tensor(s).to(self.device)
        _, probs = self.model.forward(s)
        probs = torch.squeeze(probs, 0)
        return probs[a]
    
    # Returns the action from state s by using multinomial distribution
    def get_action(self, s, useEps, useStochastic): # epsilon 0 for greedy action
        with torch.no_grad():
            s = torch.tensor(s).to(self.device)
            _, probs = self.model.forward(s)
            probs = torch.squeeze(probs, 0)

            eps = self.get_eps() if useEps else 0
            
            if random.random() >= eps:
                if useStochastic:
                    a = probs.multinomial(num_samples=1) 
                    a = a.data
                    action = a[0]
                else:
                    action = torch.argmax(probs, dim=0)
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
    def update_weight(self, Transition, isTerminal ,entropy_term = 0):
        s_t = Transition.state
        a_t = Transition.action
        s_tt = Transition.next_state
        r_tt = Transition.reward

        # get actor loss
        log_prob = torch.log(self.pi(s_t, a_t) + self.ups)
        advantage = Variable(r_tt + self.value(s_tt)*(not isTerminal) - self.value(s_t))
        actor_loss = -(advantage * log_prob)

        # get critic loss
        critic_loss = 1/2 * (r_tt + self.value(s_tt)*(not isTerminal) - self.value(s_t)).pow(2)

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
        isRender=False, 
        useTensorboard=False, 
        tensorboardTag="ActorCritic"
    ):

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
                done = False
                
                #==========================================================================
                # MAKE TRAIN DATA
                #==========================================================================

                # while not done:
                for timesteps in range(self.maxTimesteps):

                    if isRender:
                        env.render()

                    action = self.get_action(state, useEps=self.useTrainEps, useStochastic=self.useTrainStochastic)
                    next_state, reward, done, _ = self.env.step(action.tolist())

                    # trans means transition 
                    trans = Transition(state, action, next_state, reward)

                    state = next_state

                    # Train
                    if done or timesteps == self.maxTimesteps-1:
                        self.update_weight(trans, isTerminal=True)
                        break
                    else:
                        self.update_weight(trans, isTerminal=False)

                #==========================================================================
                # TEST
                #==========================================================================

                if (i_episode+1) % testPer == 0: 

                    cumulative_rewards = self.test(testSize=testSize)   
                    returns.append(cumulative_rewards)

                    #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
                    # TENSORBOARD

                    if useTensorboard:
                        writer.add_scalars("Returns", {tensorboardTag: returns[-1]}, i_episode+1)

                    #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

                    print("Episode: {0:<10} return: {1:<10}".format(i_episode + 1, returns[-1]))

        except KeyboardInterrupt:
            print("==============================================")
            print("KEYBOARD INTERRUPTION!!=======================")
            print("==============================================")

            plt.plot(range(len(returns)), returns)
        finally:
            plt.plot(range(len(returns)), returns)

        self.env.close()

    def test(self, isRender=True, testSize=10):
        
        returns = []

        for testIdx in range(testSize):
            state = self.env.reset()
            done = False
            rewards = []
            for timesteps in range(self.maxTimesteps):
                if isRender:
                    self.env.render()

                action = self.get_action(state, useEps=self.useTestEps, useStochastic=self.useTestStochastic)
                next_state, reward, done, _ = self.env.step(action.tolist())

                rewards.append(reward)
                state = next_state

                if done or timesteps == self.maxTimesteps-1:
                    break
            
            returns.append(sum(rewards))
        
        averagedReward = sum(returns) / testSize

        return averagedReward

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
