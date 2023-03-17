import sys
sys.path.append("../../../") # to import module

# 파이토치
import torch.optim as optim

# 작성자의 모듈
from grll.PG.models import ANN_V2
from grll.PG import A2C

# 환경
from grll.envs.Maze import MazeEnv_v1
trainEnv = MazeEnv_v1()
testEnv = MazeEnv_v1()
num_actions = trainEnv.num_action
num_states = trainEnv.num_obs

A2C_model = ANN_V2(num_states, num_actions)
optimizer = optim.Adam(A2C_model.parameters(), lr=1e-4)

# 작성자의 모듈 초기화
DeepQN = A2C(
    trainEnv=trainEnv,
    testEnv=testEnv,
    model=A2C_model,
    optimizer=optimizer,
    verbose=1,
    useTensorboard=True,
    maxTimesteps=int(1e3),
    actionParams={
        # for DISCRETE
        'algorithm': "greedy",  # greedy, stochastic
        'exploring': "epsilon",  # epsilon, None
        'exploringParams': {
            'start': 0.00001,
            'end': 0.00001,
            'decay': 1e10,
        },
    }, tensorboardParams={
        'logdir': "../../runs/A2C_Maze_v0",
        'tag': "Averaged Returns/ANN_V2_lr=1e-4"
    },
    nSteps=1000,
)

DeepQN.train(
        1e10,
        testPer=10000,
        testSize=5,)

DeepQN.save("../../saved_models/MazeEnv_v1/A2C_Maze_v0.obj")
