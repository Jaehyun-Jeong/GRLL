import sys
sys.path.append("../../../") # to import module

# 파이토치
import torch.optim as optim

# 작성자의 모듈
from grll.VB.models import ANN_V2
from grll.VB import ADQN

# 환경
from grll.envs.Maze import MazeEnv_v0
trainEnv = MazeEnv_v0()
testEnv = MazeEnv_v0()
num_actions = trainEnv.num_action
num_states = trainEnv.num_obs

ADQN_model = ANN_V2(num_states, num_actions)

print(ADQN_model)

optimizer = optim.Adam(ADQN_model.parameters(), lr=1e-4)

# 작성자의 모듈 초기화
AveragedDQN= ADQN(
    trainEnv=trainEnv,
    testEnv=testEnv,
    model=ADQN_model,
    optimizer=optimizer,
    verbose=1,
    useTensorboard=True,
    tensorboardParams={
        'logdir': "../../runs/ADQN_CartPole_v0",
        'tag': "Averaged Returns/ANN_V2_lr=1e-4"
    },
)

AveragedDQN.train(
        1_000_000,
        testPer=1)
