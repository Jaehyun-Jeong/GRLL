
---
# 8월 2주차
##### 정재현
##### 이아영 (모든 이미지 작업)
---

# 1. stable-baselines3와 작성자 모듈의 A2C 알고리즘 속도비교

## 1.1. 비교환경

## 1.2. CartPole-v0 환경에서의 비교

## 1.3. 결론

## 1.4. 작성자 모듈의 개선점

**stable-baselines3의 소스코드를 확인했다. 그리고 다음과 같은 차이점을 발견했다.**

### 1.4.1. Advantage Normalization

**Advantage Normalization을 사용할 수 있는 코드가 존재했다.**<br/>
![](normalize_advantage.png)<br/>
<https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/a2c/a2c.py>

간단한 계산이기 때문에 바로 구현 가능한다.

### 1.4.2. Entropy Loss Term

**학습 안정화를 위한 Entropy Loss를 사용한다.**
![](entropy_loss.png)<br/>
<https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/a2c/a2c.py>

Entropy Loss Term을 사용하는 방법은 여러가지 있다. 작성자의 지식 부족으로 아직 구현할 수 없다.

### 1.4.3. Gradient Clipping

**stable-baselines3는 Gradient Exploding 문제를 해결하기 위한 Gradient Clipping을 지원한다.**

stable-baselines3에서는 상수 값을 사용하는 Gradient Clippint, 그리고 Value Clipping, Norm Clipping을 지원한다.

**작성자의 모듈도 사용할 수 있도록 코드 수정을 고려하고 있다.**

# 2. 복수의 행동을 위한 힌트 발견

![](continuous_control.png)<br/>
*Volodymyr Mnih, Adria Puigdomenech Badia, Mehdi Mirza, Alex Graves, Timothy Lillicrap, Tim Harley, David Silver, and Koray Kavukcuoglu. Asynchronous methods for deep reinforcement learning. In International Conference on Machine Learning, pages 1928–1937, 2016.*

A3C 논문에서는 MuJoCo Physics Simulator문제를 위와 같은 방법으로 해결했다.<br/>
**즉, 뉴럴넷이 2개의 값(평균, 표준편차)을 행동 개수만큼 출력하도록 만들었다.**<br/><br/>

> 예를 들어, 거미를 앞으로 나아가게 하는 목표를 가지고 있고, 움직일 수 있는 관절의 수를 8개라고 가정하자.<br/>
> 그러면, A3C(A2C도 동일) 뉴럴넷은 16개의 출력값을 가진다. (평균 8개, 표준편차 8개) 
