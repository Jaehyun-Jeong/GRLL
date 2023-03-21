import sys
sys.path.append("../../../") # to import module

# 파이토치
import torch.optim as optim

# 작성자의 모듈
from grll.VB.models import ANN_Maze
from grll.VB import DQN

# 환경
from grll.envs.Maze import MazeEnv_v2

trainEnv = MazeEnv_v2(exploring_starts=True)
testEnv = MazeEnv_v2(exploring_starts=False)
num_actions = trainEnv.num_action
num_states = trainEnv.num_obs

lr = 0.002
discount = 0.5

DQN_model = ANN_Maze(num_states, num_actions)
optimizer = optim.Adam(DQN_model.parameters(), lr=lr)

# 작성자의 모듈 초기화
DeepQN = DQN(
    trainEnv=trainEnv,
    testEnv=testEnv,
    model=DQN_model,
    optimizer=optimizer,
    verbose=1,
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
        'logdir': "../../runs/DQN_Maze_v2_bests",
        'tag': f"Maze_v2_candidates/lr={lr}_discount={discount}"
    },
    epoch=4,
    gradientStepPer=4,
    trainStarts=1000,
)

DeepQN.train(
        10_000_000,
        testPer=10,
        testSize=1,)
