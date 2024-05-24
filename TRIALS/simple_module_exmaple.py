import sys
sys.path.append("../") # to import module

import torch.optim as optim

# 만들어진 모듈과 기본으로 제공하는 뉴럴넷 모델을 임포트
from grll.PG.models import ANN_V2
from grll.PG import A2C

# 환경
import gymnasium as gym
env = gym.make('CartPole-v0')

# 뉴럴넷을 생성
num_actions = env.action_space.n
num_states = env.observation_space.shape[0]
A2C_model = ANN_V2(num_states, num_actions)

# Optimizer를 생성
optimizer = optim.Adam(A2C_model.parameters(), lr=1e-4)

# 강화학습 클래스 초기화
advantage_AC = A2C(
    env=env,
    model=A2C_model,
    optimizer=optimizer,
)

# 학습진행
advantage_AC.train(trainTimesteps=1000000)

# 클래스 저장
ADeepQLearning.save("./saved_models/test.obj")
