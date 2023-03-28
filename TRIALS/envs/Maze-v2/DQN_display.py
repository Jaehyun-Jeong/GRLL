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
        maze=copy(maze),
        displayMode=True,
        )
# testEnv.blocks = trainEnv.blocks

lr = float(sys.argv[1])
discount = float(sys.argv[2])

num_actions = trainEnv.num_action
num_states = trainEnv.num_obs

DQN_model = ANN_Maze(num_states, num_actions)
optimizer = optim.Adam(DQN_model.parameters(), lr=lr)

# 작성자의 모듈 초기화
DeepQN = DQN(
    trainEnv=trainEnv,
    testEnv=testEnv,
    model=DQN_model,
    optimizer=optimizer,
)

DeepQN.load(f"../../saved_models/MazeEnv_v2/DQN_Maze_v2_lr={lr}_discount={discount}")
policy_diagram(testEnv, DeepQN)
DeepQN.isRender['test'] = True

DeepQN.test(testSize=1)
