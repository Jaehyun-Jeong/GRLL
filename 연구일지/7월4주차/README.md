
---
# 7월 4주차
##### 정재현
##### 이아영 (게임 이미지 작업)
---

# 1. Actor Critic for Continuous Action Space

![](ActorCritic_continuous.png)<br/>
*[Continuous Action Space Actor Critic Tutorial](https://www.youtube.com/watch?v=kWHSH2HgbNQ&t=153s&ab_channel=SkowstertheGeek)*

# 2. 자동차 환경 수정

## 2.1. 학습 가능하도록 코드를 수정

**다음 코드는 사용 예시이다.**<br/>
```python
import sys
sys.path.append("../") # to import module

# PyTorch
import torch
import torch.optim as optim

# import model
from module.ActorCritic.models import ANN_V1
from module.ActorCritic import onestep_ActorCritic

# 환경
from module.envs.CarRacing import RacingEnv_v0
env = RacingEnv_v0()

# 뉴럴넷 모델과 옵티마이저
num_actions = env.num_actions
num_states = env.num_obs
ActorCritic_model = ANN_V1(num_states, num_actions).to(device)
optimizer = optim.Adam(ActorCritic_model.parameters(), lr=0.1e-3)

# 강화학습 모듈 초기화
onestep_AC = onestep_ActorCritic(
    env=env,
    model=ActorCritic_model,
    optimizer=optimizer,
)

# 학습
onestep_AC.train(maxEpisodes=1000)

# 저장
onestep_AC.save("./saved_models/onestep_ActorCritic_RacingEnv_v0.obj")
```

## 2.2 렌더링을 설정할 수 있다.

**렌더링은 학습의 속도를 줄인다. 따라서 렌더링 여부를 선택 할 수 있도록 코드를 수정했다.**

## 2.3 Exploring Starts 요소를 넣었다.
