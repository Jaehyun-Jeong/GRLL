import sys
sys.path.append("../../../") # to import module

# 파이토치
import torch.optim as optim
import torch.nn as nn

# 작성자의 모듈
from grll.VB import DQN
from grll.VB.models import ANN_V2

# 환경
from grll.envs.Car import CarEnv_v0
trainEnv = CarEnv_v0(carSize=(30, 30), difficulty=4)
testEnv = CarEnv_v0(carSize=(30, 30), difficulty=4)

num_actions = trainEnv.num_action
num_states = trainEnv.num_obs

lr = float(sys.argv[1])
discount = float(sys.argv[2])

DQN_model = ANN_V2(num_states, num_actions)
optimizer = optim.Adam(DQN_model.parameters(), lr=lr)


# 작성자의 모듈 초기화
DeepQN = DQN(
    trainEnv=trainEnv,
    testEnv=testEnv,
    model=DQN_model,
    optimizer=optimizer,
    verbose=1,
    maxTimesteps=10000,
    maxMemory=10000,
    numBatch=526,
    discount=discount,
    actionParams={
        # for DISCRETE
        'algorithm': "greedy",  # greedy, stochastic
        'exploring': "epsilon",  # epsilon, None
        'exploringParams': {
            'start': 1,
            'end': 0.01,
            'decay': 30000,
        },
    },
    useTensorboard=False,
    tensorboardParams={
        'logdir': "../../runs/DQN_Car_v0",
        'tag': f"Averaged Returns/ANN_V2_lr={lr}_discount={discount}"
    },
    useCheckpoint=True,
    checkpointParams={
        'savePath': "../../saved_models/CarEnv_v0/DQN_Car_v0.obj",
        'metric': 'episode'
    },
    epoch=4,
    gradientStepPer=1,
    trainStarts=1000,
    updateTargetPer=10000,
)

DeepQN.train(
    200000,
    testPer=10,
    testSize=1
)
