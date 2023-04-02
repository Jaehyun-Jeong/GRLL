import sys
sys.path.append("../../../") # to import module

# 파이토치
import torch.optim as optim

# 작성자의 모듈
from grll.VB.models import ANN_Maze
from grll.VB import DQN

# 환경
from grll.envs.Maze import MazeEnv_v1
trainEnv = MazeEnv_v1()
testEnv = MazeEnv_v1()
num_actions = trainEnv.num_action
num_states = trainEnv.num_obs

DQN_model = ANN_Maze(num_states, num_actions)
optimizer = optim.Adam(DQN_model.parameters(), lr=1e-4)

# 작성자의 모듈 초기화
DeepQN = DQN(
    trainEnv=trainEnv,
    testEnv=testEnv,
    model=DQN_model,
    optimizer=optimizer,
    verbose=1,
    useTensorboard=True,
    maxTimesteps=int(1e3),
    maxMemory=num_states*8,  # 8 times of state size
    numBatch=32,
    actionParams={
        # for DISCRETE
        'algorithm': "greedy",  # greedy, stochastic
        'exploring': "epsilon",  # epsilon, None
        'exploringParams': {
            'start': 0.0001,
            'end': 0.0001,
            'decay': 1e10,
        },
    }, tensorboardParams={
        'logdir': "../../runs/DQN_Maze_v0",
        'tag': "Averaged Returns/ANN_V2_lr=1e-4"
    },
)

DeepQN.train(
        1e10,
        testPer=1000,
        testSize=5,)

DeepQN.save("../../saved_models/MazeEnv_v1/DQN_Maze_v0.obj")
