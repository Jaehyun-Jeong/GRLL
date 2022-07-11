import sys
sys.path.append("../") # to import module

# 만들어진 모듈과 기본으로 제공하는 뉴럴넷 모델을 임포트
from module.ActorCritic.models import ANN_V2
from module.ActorCritic import onestep_ActorCritic

# Optimizer를 결정하기 위해 임포트
import torch.optim as optim

# 환경
import gym
env = gym.make('CartPole-v0')

# 뉴럴넷을 생성
num_actions = env.action_space.n
num_states = env.observation_space.shape[0]
ActorCritic_model = ANN_V2(num_states, num_actions)

# Optimizer를 생성
optimizer = optim.Adam(ActorCritic_model.parameters(), lr=0.1e-3)

# 강화학습 클래스 초기화
AC = onestep_ActorCritic(
    env = env,
    model = ActorCritic_model,
    optimizer = optimizer
)

# 불러오기
AC.load("./saved_models/test.obj")

# 테스트
AC.isRender['test'] = True # 환경을 출력하도록 변경
AC.test(testSize=10)

