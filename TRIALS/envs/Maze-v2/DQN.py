import sys
sys.path.append("../../../") # to import module

# 파이토치
import torch.optim as optim

# 작성자의 모듈
from grll.VB.models import ANN_Maze
from grll.VB import DQN

# 환경
from grll.envs.Maze import MazeEnv_v2
from grll.envs.Maze.utils import policy_diagram

# 미로 정의
import numpy as np
from copy import copy
'''
maze = np.array([
    [0, 0, 1, 1, 0, 0],
    [0, 0, 0, 1, 1, 0],
    [0, 0, 1, 0, 0, 0],
    [0, 1, 1, 0, 1, 0],
    [0, 1, 0, 0, 1, 0],
    [0, 0, 0, 0, 1, 0],
])
'''
maze =  np.array([
    [ 0.,  1.,  0.,  0.,  0.,  0.,  0.],
    [ 0.,  0.,  0.,  1.,  1.,  0.,  1.],
    [ 1.,  1.,  1.,  0.,  0.,  0.,  1.],
    [ 0.,  0.,  0.,  0.,  1.,  1.,  0.],
    [ 0.,  1.,  1.,  1.,  0.,  0.,  0.],
    [ 0.,  1.,  0.,  0.,  0.,  0.,  0.],
    [ 0.,  0.,  0.,  1.,  0.,  0.,  0.]
])

trainEnv = MazeEnv_v2(
    exploringStarts=True,
    mazeSize=(7, 7),
    maze=copy(maze)
)
testEnv = MazeEnv_v2(
    exploringStarts=False,
    mazeSize=(7, 7),
    maze=copy(maze)
)
# testEnv.blocks = trainEnv.blocks

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
    verbose=1,
    useTensorboard=False,
    maxTimesteps=int(1e100),
    maxMemory=10000, # 8 times of state size
    numBatch=526,
    discount=discount,
    actionParams={
        # for DISCRETE
        'algorithm': "greedy",  # greedy, stochastic
        'exploring': 'epsilon',  # epsilon, None
        'exploringParams': {
            'start': 1,
            'end': 0.01,
            'decay': 30000,
        },
    }, tensorboardParams={
        'logdir': "../../runs/DQN_Maze_v2",
        'tag': f"Averaged Returns/ANN_Maze_lr={lr}_discount={discount}"
    },
    epoch=4,
    gradientStepPer=1,
    trainStarts=1000,
    updateTargetPer=10000,
)

DeepQN.train(
        50000,
        testPer=10,
        testSize=1,)
policy_diagram(testEnv, DeepQN)
