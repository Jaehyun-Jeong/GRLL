# GRLL

| **Algorithm** | **Discrete Action Space** | **Continuous Action Space** | 
| ------------- | ------------------------- | --------------------------- |
| A2C | :heavy_check_mark: | :heavy_check_mark: |
| REINFORCE | :heavy_check_mark: | :heavy_check_mark: |
| DQN | :heavy_check_mark: | :x: |
| ADQN | :heavy_check_mark: | :x: |

## Install

## Usage

아래는 OpenAI의 gym으로 테스트한 예제이다.

```python
import torch.optim as optim
from GRLL.PG.models import ANN_V2
from GRLL.PG import A2C
import gym

env = gym.make('CartPole-v0')

num_actions = env.action_space.n
num_states = env.observation_space.shape[0]
A2C_model = ANN_V2(num_states, num_actions)

optimizer = optim.Adam(A2C_model.parameters(), lr=1e-4)

advantage_AC = A2C(
    env=env,
    model=A2C_model,
    optimizer=optimizer,
)

advantage_AC.train(trainTimesteps=1000000)
```

만약 다른 알고리즘을 사용하고 싶다면 다음과 같이 작성하면 된다.<br/>
```python
from GRLL.PG import REINFORCE
"""
또는
from GRLL.VB import DQN
from GRLL.VB import ADQN
"""
```

## Custom Environment

만약 pygame 모듈을 다운로드 했다면 다음 두 가지 환경을 사용할 수 있다.

### RacingEnv

![](/static/RacingEnv.png)

RacingEnv_v0: 5개의 센서 길이값을 state로 받고, 오른쪽, 왼쪽, 액셀, 브레이크 4개의 행동을 가진다.<br/>

### MazeEnv

![](/static/MazeEnv.png)

[NeuralNine](https://www.youtube.com/watch?v=Cy155O5R1Oo&t=527s&ab_channel=NeuralNine)

MazeEnv_v0: 맵 전체의 벡터 정보를 state로 받고, 동서남북으로 움직이는 4개의 행동을 가진다.<br/>
