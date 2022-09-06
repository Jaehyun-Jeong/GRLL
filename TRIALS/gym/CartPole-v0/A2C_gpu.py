import sys
sys.path.append("../../../") # to import module

from datetime import datetime

# 시작 시간
startTime = datetime.now()

# 파이토치
import torch
import torch.optim as optim

# 작성자의 모듈
from module.PG.models import ANN_V3
from module.PG import A2C

# 환경
import gym
env = gym.make('CartPole-v0')
num_actions = env.action_space.n
num_states = env.observation_space.shape[0]

A2C_model = ANN_V3(num_states, num_actions)
optimizer = optim.Adam(A2C_model.parameters(), lr=1e-4)

# 작성자의 모듈 초기화
advantage_AC = A2C(
    env=env,
    model=A2C_model,
    device=torch.device('cuda'),
    optimizer=optimizer,
    verbose=1,
    policy={
        'train': 'stochastic',
        'test': 'greedy',
    },
)

# 모듈 초기화에 걸린 시간
print(f"Init Time: {datetime.now() - startTime}")

# 학습이 시작되는 시간
startTrainTime = datetime.now()

advantage_AC.train(
        trainTimesteps=1000000,
        testSize=10)

# 학습이 끝나는 시간
print(f"Train Time: {datetime.now() - startTrainTime}")

# 성능 측정을 위한 테스트
print(advantage_AC.test(testSize=10))
print("=================================================")
