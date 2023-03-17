import sys
sys.path.append("../../../") # to import module

# 파이토치
import torch.optim as optim

# 작성자의 모듈
from grll.VB.models import ANN_Maze
from grll.VB import DQN

# 환경
from grll.envs.Maze import MazeEnv_v2
trainEnv = MazeEnv_v2()
testEnv = MazeEnv_v2()
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
    maxTimesteps=int(1e100),
    maxMemory=num_states*8,  # 8 times of state size
    numBatch=16,
    actionParams={
        # for DISCRETE
        'algorithm': "stochastic",  # greedy, stochastic
        'exploring': None,  # epsilon, None
    }, tensorboardParams={
        'logdir': "../../runs/DQN_Maze_v0",
        'tag': "Averaged Returns/ANN_Maze_lr=1e-4"
    },
    epoch=8,
)

DeepQN.train(
        1e10,
        testPer=10000,
        testSize=1,)

DeepQN.save("../../saved_models/MazeEnv_v2/DQN_Maze_v2.obj")
