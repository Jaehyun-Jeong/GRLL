import sys
sys.path.append("../../../") # to import module

# 파이토치
import torch.optim as optim
import torch.nn as nn

class ANN_Car(nn.Module):
    def __init__(self, inputs, outputs):
        super(ANN_Car, self).__init__()

        self.layers = nn.Sequential(
                nn.Linear(inputs, 256),
                nn.LeakyReLU(),
                nn.BatchNorm1d(256),
                nn.Linear(256, 256),
                nn.LeakyReLU(),
                nn.BatchNorm1d(256),
                nn.Linear(256, outputs),
            )

    def forward(self, x):

        value = self.layers(x)

        return value

# 작성자의 모듈
from grll.VB import DQN

# 환경
from grll.envs.Car import CarEnv_v0
trainEnv = CarEnv_v0(difficulty=3)
testEnv = CarEnv_v0(difficulty=3)
num_actions = trainEnv.num_action
num_states = trainEnv.num_obs

DQN_model = ANN_Car(num_states, num_actions)
optimizer = optim.Adam(DQN_model.parameters(), lr=1e-4)


# 작성자의 모듈 초기화
DeepQN = DQN(
    trainEnv=trainEnv,
    testEnv=testEnv,
    model=DQN_model,
    optimizer=optimizer,
    verbose=1,
    maxTimesteps=10000,
    maxMemory=10000,
    numBatch=16,
    discount=0.9,
    actionParams={
        # for DISCRETE
        'algorithm': "greedy",  # greedy, stochastic
        'exploring': "epsilon",  # epsilon, None
        'exploringParams': {
            'start': 0.95,
            'end': 0.05,
            'decay': 10000,
        },
    }, 
    useTensorboard=True,
    tensorboardParams={
        'logdir': "../../runs/DQN_Car_v0",
        'tag': "Averaged Returns/ANN_V2_lr=1e-4"
    },
    trainStarts=10000,
)

DeepQN.train(
        1_000_000_000,
        testPer=10000,
        testSize=1,)

DeepQN.save("../../saved_models/CarEnv_v0/DQN_Car_v0.obj")
