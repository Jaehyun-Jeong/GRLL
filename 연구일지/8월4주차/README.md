
---
# 8월 2주차
##### 정재현
##### 이아영 (모든 이미지 작업)
---

# 1. 미로 환경 MazeEnv_v0 제작

![](MazEnv_v0_running.png)<br/>
*실행 중의 사진*

**다음은 사용 예시 코드이다.**

```python
from module.envs.MazeEnv_v0
from random import choice 

env = MazeEnv_v0()

while True:
    env.render()  # 렌더링 여부를 결정 가능하다.
    action = choice([0, 1, 2, 3])  # 동, 서, 남, 북 중에 랜덤으로 선택
    next_state, reward, done, action = env.step(action)  # 한 번의 Timestep
```
