import sys
sys.path.append("../../../") # to import module

# 파이토치
import torch.optim as optim

# 작성자의 모듈
from grll.VB.models import ANN_V2
from grll.VB import DQN

# 환경
from grll.envs.Car import CarEnv_v0
trainEnv = CarEnv_v0(carSize=(30, 30), difficulty=4)
testEnv = CarEnv_v0(carSize=(30, 30), difficulty=4)
num_actions = trainEnv.num_action
num_states = trainEnv.num_obs

DQN_model = ANN_V2(num_states, num_actions)
optimizer = optim.Adam(DQN_model.parameters(), lr=1e-4)

# 작성자의 모듈 초기화
DeepQN = DQN(
    trainEnv=trainEnv,
    testEnv=testEnv,
    model=DQN_model,
    optimizer=optimizer,
    verbose=1,
    useTensorboard=False,
    tensorboardParams={
        'logdir': "../../runs/DQN_Car_v0",
        'tag': "Averaged Returns/ANN_V2_lr=1e-4"
    },
)

DeepQN.load("../../saved_models/CarEnv_v0/DQN_Car_v0.obj")
DeepQN.isRender['test'] = True

DeepQN.test(testSize=1)
