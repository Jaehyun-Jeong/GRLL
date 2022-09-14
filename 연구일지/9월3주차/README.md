
---
# 9월 3주차
##### 정재현
##### 이아영 (모든 이미지 작업)
---

# 1. OpenAI의 Box와 같은 class를 작성

OpenAI의 Box, Discrete와 같은 action space를 참고하여 모듈 내부의 class를 작성했다.<br/>
> Action Space란 에이전트가 행동 가능한 수학적 공간을 의미한다.<br/>
> 예를 들어, 자동차가 꺽울 수 있는 각도를 -180, 180도 라고 하면, 이 환경의 Action Space는 -180과 180사이의 실수이다.<br/>
> 그리고 질문에 대답하는 챗봇을 만들었다고 가정하면, 언어의 모든 조합이 Action Space이다.<br/><br/>

지금은 우선 Discrete, Continuous만 가능하도록 코드를 작성했으며 코드는 ./module/utils/ActionSpace.py 를 참고하면 된다.<br/>

# 2. 여러 sheduling을 지원하고자 sheduling class 작성

작성자의 모듈은 Epsilon-Greedy Policy에서의 Epsilon의 scheduling만 지원하고 있었다. 하지만, learning rate scheduling, 그리고 optimizer의 momentum과 scaling paramter 또한 scheduling하는 등, 여러 곳에서 sheduling 알고리즘이 사용된다. 따라서 이를 따로 분리시킬 필요가 있다고 생각했다.<br/><br/>

**우선 linear과 exponential scheduling 코드를 작성했다. 코드는 ./module/utils/scheduling.py를 참고하면 된다.**

# 3. exploring 분리

**이는 아직 작업중이다.**<br/><br/>

**빨리 테스트를 진행하기 위해 다음과 같은 두 가지 exploring만 지원하도록 작성중이다.**<br/>
* Discrete<br/>
* * Epsilon Exploring<br/>
* Continuous<br/>
* * normalized noise<br/><br/>

**코드는 ./module/utils/exploring.py를 참고하면 된다.**

# 4. Policy를 만들기 위한 코드 리팩토링과 캡슐화

**이는 아직 작업중이다.**<br/><br/>

지금까지는 알고리즘 안에 모든 방법론을 적용시킬 수 있도록 했다. 하지만 조사를 하면 할 수록 여러 알고리즘이 발견되고, 현재 방식의 한계를 발견했다..**따라서 다음과 같은 기능들은 세분화 하고자 작업중이다.**<br/><br/>

* State Value와 Action Value만을 계산하는 ActionValue, StateValue class 작성<br/>
* Action Space와 Acion Value를 받아서 행동을 출력해주는 Policy 작성<br/>

# 5. 다음 주 계획

저번 주 과제를 이어서 하고자 한다.<br/>

