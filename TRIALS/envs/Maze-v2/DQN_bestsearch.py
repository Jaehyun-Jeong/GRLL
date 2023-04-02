import sys
sys.path.append("../../../") # to import module

# 파이토치
import torch.optim as optim

# 작성자의 모듈
from grll.VB.models import ANN_Maze
from grll.VB import DQN

# 환경
from grll.envs.Maze import MazeEnv_v2

# 미로 정의
import numpy as np
maze = np.array([
    [ 0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
    [ 0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.],
    [ 0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.],
    [ 1.,  1.,  0.,  1.,  1.,  0.,  1.,  0.,  0.,  0.],
    [ 0.,  0.,  1.,  0.,  1.,  0.,  1.,  1.,  1.,  0.],
    [ 0.,  0.,  1.,  0.,  1.,  0.,  0.,  0.,  0.,  0.],
    [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
    [ 0.,  0.,  0.,  0.,  0.,  0.,  1.,  1.,  1.,  1.],
    [ 0.,  1.,  1.,  1.,  1.,  1.,  0.,  0.,  0.,  0.],
    [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.]
])

trainEnv = MazeEnv_v2(
        exploringStarts=True,
        mazeSize=(10, 10),
        maze=maze
        )
testEnv = MazeEnv_v2(
        exploringStarts=False,
        mazeSize=(10, 10),
        maze=maze
        )

num_actions = trainEnv.num_action
num_states = trainEnv.num_obs

lr = float(sys.argv[1])
discount = float(sys.argv[2])

DQN_model = ANN_Maze(num_states, num_actions)
optimizer = optim.Adam(DQN_model.parameters(), lr=lr)

# 작성자의 모듈 초기화
DeepQN = DQN(
    trainEnv=trainEnv,
    testEnv=testEnv,
    model=DQN_model,
    optimizer=optimizer,
    verbose=0,
    useTensorboard=True,
    maxTimesteps=int(1e100),
    maxMemory=10000, # 8 times of state size
    numBatch=12,
    discount=discount,
    actionParams={
        # for DISCRETE
        'algorithm': "greedy",  # greedy, stochastic
        'exploring': 'epsilon',  # epsilon, None
        'exploringParams': {
            'start': 0.9,
            'end': 0.001,
            'decay': 1000,
        },
    }, tensorboardParams={
        'logdir': "../../runs/DQN_Maze_v2",
        'tag': f"Averaged Returns/ANN_Maze_lr={lr}_discount={discount}"
    },
    epoch=4,
    gradientStepPer=4,
    trainStarts=1000,
)

DeepQN.train(
        10000,
        testPer=10,
        testSize=1,)
