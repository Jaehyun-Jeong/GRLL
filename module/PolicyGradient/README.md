# ActorCritic

![](https://firebasestorage.googleapis.com/v0/b/aing-biology.appspot.com/o/sutton_barto_reinforcement_learning%2Fchapter13%2F02.PNG?alt=media&token=19f2ffc4-aaac-45b2-b537-eb7c02231abd)

*출처: Richard Sutton and Andrew Barto. Reinforcement Learning: An Introduction. MIT Press, 1998.*

## 1) Value Based Method

Policy는 상태(state)들의 집합에서 행동의 확률분포(distribution)로 가는 함수이다.

그리고 상태 $s$에서 Policy $\pi$를 따라가는 State Value Function은 다음과 같이 정의한다.

$ v_\pi (s) = \mathbb{E}{\pi} \left[ G_{t} | S_{t} = s \right] = \mathbb{E}{\pi} \left[ \displaystyle\sum_{k=0}^{\infty} \gamma^k R_{t+k+1} | S_{t} = s \right] \, \text{ for all } s \in \mathcal{S} $

또한, 상태 $ s $에서 행동 $ a $를 선택하고 Policy $ \pi $를 따라가는 Action Value Function은 다음과 같다.

$q_\pi (s, a) = \mathbb{E}{\pi} \left[ G_{t} | S_{t} = s, A_{t} = a \right] = \mathbb{E}{\pi} \left[ \displaystyle\sum_{k=0}^{\infty} \gamma^k R_{t+k+1} | S_{t} = s, A_{t} = a \right] \ \ \text{ for all } s \in \mathcal{S}, a \in \mathcal{A}(s)$

**위같은 함수들을 Value Function이라 하고, 에이전트가 Value Function 기반으로 행동을 결정할 경우 Value Based Method라 한다.**

## 2) Actor-Critic의 의미

위의 의사코드를 보면 알 수 있듯이 Actor-Critic Method는 두 가중치 $ w, \theta $를 업데이트한다.

**이같이, Policy(Actor)와 Value Function(Critic)을 동시에 학습하는 방법을 Actor-Critic Method라 한다.**

> 보면 $ \hat{v} $과 같이 표현했는데, (hat)은 추정값을 의미한다. 그리고 추정값을 사용하는 이유는 환경에서 임의의 상태에 대해 그 Value를 알 수 없기 때문이다.

## 3) Critic과 Variance

위에서 이미 설명했듯이, Policy는 상태에서 행동의 확률로 가는 함수이다.따라서 만약 행동 $a_{1}$이 행동 $a_{2}$보다 '바람직하다'면 $a_{1}$의 확률이 더 높게 나올것이다.

> 여기서 '바람직하다'는 큰 return값을 얻을 수 있음을 의미한다.

이같이, Policy를 업데이트 하는데는 각 행동에 대한 상대적인 수정이 필요하다.

**즉, REINFORCE 알고리즘과 큰 Variance를 가지는 값($G_{t}$)으로 업데이트할 필요가 없다.**

> 큰 값으로 업데이트를 진행하면 Variance가 커지는 이유는 다음 예제를 보면 알 수 있다.

![](https://miro.medium.com/max/1400/1*3r6GvYe9Xm0xWrmNIoatzw.png)

*출처: [Jerry Liu’s post to “Why does the policy gradient method have high variance”](https://www.quora.com/unanswered/Why-does-the-policy-gradient-method-have-a-high-variance)*

이제 $ G_{t} $의 평균값 $ \bar{G} $을 알고있다고 가정하자. 이때, $ G_{t} - \bar{G} $로 업데이트를 하면 Variance를 줄일 수 있다.

**이같이, Variance를 줄이기 위해 사용하는 $ \bar{G} $와 같은 항을 baseline이라 한다.**

이러한 관점에서 Actor-Critic 의사코드에서 $\ G_{t} $대신에 $\ \delta $와 같은 표현을 사용한 이유는 다음과 같다.

---

#### Equation 1

$R_{t+1} + \gamma \hat{v} (s_{t+1}, w) $

$= R_{t+1} + \gamma \mathbb{E} _{\pi} [G_{t+1} | s_{t+1}] $

$= R_{t+1} + \gamma \mathbb{E}_{\pi} \left[\sum_{k=0}^{\infty} \gamma^k R_{t+k+2} | s_{t+1} \right] $

$\approx R_{t+1} + \gamma \left( R_{t+2} + R_{t+2} + \cdots \right) $

$= G_t $

---

#### Equation 2

$ R_{t+1} + \gamma \hat{v} (s_{t+1}, w) $

$ = \hat{q} (s_t, a_t, w) $

---

#### Equation 3

$ \delta = \hat{q} (s_t, a_t, w) - \hat{v} (s_t, w) $

$ = \hat{q} (s_t, a_t, w) - \displaystyle\sum_a \pi(a | s_t) \hat{q} (s_t, a, w) $

* $\pi (a, | s_{t}) $ : 상태 $s_t $에서 행동 $a $를 선택할 확률

---

**Equation 1은 $G_t$대신에 $R_{t+1} + \gamma \hat{v} (s_{t+1}, w)$를 사용할 수 있는 이유를 설명한다.**

**Equation 2는 $R_{t+1} + \gamma \hat{v} (s_{t+1}, w)$와 Action Value Function이 같음을 설명한다.**

**Equation 3는 $\bar{G}$ 대신에 $\hat{v} (s_t, w)$를 사용할 수 있는 이유를 설명한다.**

**이렇게 만들어진 $\delta$를 Advantage라 한다.**
