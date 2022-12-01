import sys
sys.path.append("../../../") # to import module

# 파이토치
import torch.optim as optim

# 작성자의 모듈
from grll.PG.models import ANN_V2
from grll.PG import A2C

# 환경
import gym
env = gym.make('Ant-v2')
num_actions = env.action_space.shape[0]
num_states = env.observation_space.shape[0]

A2C_model = ANN_V2(num_states, num_actions)
optimizer = optim.Adam(A2C_model.parameters(), lr=1e-4)

# 작성자의 모듈 초기화
advantage_AC = A2C(
    env=env,
    model=A2C_model,
    optimizer=optimizer,
    verbose=1,
    useTensorboard=True,
    actionParams={
        'algorithm': 'plain',
        'exploring': 'normal',
        'exploringParams': {
            'mean': 0,
            'sigma': 0,
        },
    },
    tensorboardParams={
        'logdir': "../../runs/A2C_Ant_v2",
        'tag': "Averaged Returns/ANN_V2_lr=1e-4"
    },
    nSteps=10,
)

advantage_AC.train(trainTimesteps=int(1e8))
