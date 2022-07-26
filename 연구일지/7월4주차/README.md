
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

**다음과 같이 모듈 초기화를 하면 된다.**<br/>
```python
# 강화학습 모듈 초기화
onestep_AC = onestep_ActorCritic(
    env=env,
    model=ActorCritic_model,
    optimizer=optimizer,
    isRender={
        'train': False,
        'test': True,
    }
)
```
*학습시에는 렌더링 하지 않고, 테스트 때는 렌더링 한다.*

### 2.2.1.렌더링에 따른 속도 테스트는 다음과 같이 진행했다.

1. 렌더링의 유무에 관계없이 둘 다 onestep-ActorCritic 알고리즘을 사용했다.
2. 속도를 정확하게 측정하기 위해 테스트 에피소드는 진행하지 않았다.
3. 학습은 1000에피소드까지 진행되었다.
4. 학습의 정도에 따라 에피소드의 길이가 달라질 수 있다. 따라서 실제 비교값과 오차가 예상된다.

**학습이 완료되는데, 렌더링 한 경우는 50분 39초가 걸렸고, 렌더링이 없는 경우는 39분 37초가 걸렸다.**<br/>
**즉, 1000개의 에피소드를 렌더링 없이 진행할 경우 약, 10분 정도 더 빨리끝났다.**

## 2.3 Exploring Starts 요소를 넣었다.

**다음 사진에서 빨간 점이 있는 부분 중 렌덤으로 시작한다.**<br/>
![](Exploring_Starts.png)<br/>

이는 에이전트가 더 다양한 경험을 하게 만듦으로써 원활한 학습을 도와준다.
