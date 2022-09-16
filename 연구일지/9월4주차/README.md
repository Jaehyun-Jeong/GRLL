
---
# 9월 4주차
##### 정재현
##### 이아영 (모든 이미지 작업)
---

# 9월 15일

## 1. Policy Gradient의 ActionValue와 StateValue를 계산하는 class 작성

```python
from typing import Union, Dict

import numpy as np

# PyTorch
import torch

# module
from module.Policy import DiscretePolicy, ContinuousPolicy
from module.utils.ActionSpace import ActionSpace


class Value():

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer,
        actionSpace: ActionSpace,
        actionParams: Dict[str, Union[int, float, Dict]] = {

            # for DISCRETE

            'algorithm': "greedy",  # greedy, stochastic
            'exploring': "epsilon",  # epsilon, None
            'exploringParams': {
                'start': 0.99,
                'end': 0.0001,
                'decay': 10000
            }
        },
        clippingParams: Dict[str, Union[int, float]] = {
            'pNormValue': 2,
            'maxNorm': 1,
        },
    ):

        # Initialize Parameter
        self.actionSpace = actionSpace
        self.stepsDone = 0

        # Set policy
        if self.actionSpace.actionType == 'Discrete':
            self.policy = DiscretePolicy(**actionParams)
        elif self.actionSpace.actionType == 'Continuous':
            self.policy = ContinuousPolicy(**actionParams)
        else:
            raise ValueError(
                "actionType only for Discrete and Continuous Action")

    # Update Weights
    def step(self, loss):

        # Calculate Gradient
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=self.clippingParams['maxNorm'],
                norm_type=self.clippingParams['pNormValue'],
                )

        # Backpropagation and count steps
        self.optimizer.step()
        self.stepsDone += 1

    # Returns a value of the state
    # (state value function in Reinforcement learning)
    def StateValue(
            self,
            s: torch.Tensor,
            ) -> torch.Tensor:

        value, _ = self.model.forward(s)

        return value

    # Get Action Value from state
    def ActionValue(
            self,
            s: Union[torch.Tensor, np.ndarray],
            ) -> torch.Tensor:

        s = torch.Tensor(s).to(self.device).unsqueeze(0)
        _, ActionValue = self.model.forward(s)
        ActionValue = ActionValue.squeeze(0)

        return ActionValue

    # In Reinforcement learning,
    # pi means the function from state space to action probability distribution
    # Returns probability of taken action a from state s
    def pi(
            self,
            s: torch.Tensor,
            a: torch.Tensor) -> torch.Tensor:

        a = a.unsqueeze(dim=-1)

        _, probs = self.model.forward(s)
        actionValue = torch.gather(torch.clone(probs), 1, a).squeeze(dim=1)

        return actionValue

    # Get Action from State s
    @torch.no_grad()
    def get_action(
            self,
            s: Union[torch.Tensor, np.ndarray],
            ) -> torch.Tensor:

        ActionValue = self.value(s)

        return self.policy(
                ActionValue,
                self.stepsDone
                )
```

## 1. Gym의 ActionSpace인 Box와 Discrete를 모듈 내부의 ActionSpace로 변환하는 함수 작성

```python
from gym.spaces import Box, Discrete
import numpy as np

from module.utils.ActionSpace import ActionSpace


def fromDiscrete(
        space: Discrete
        ):

    # Biggiest index of action is space.n - 1
    # Because space.n is a size of action space
    high = np.array([space.n-1], dtype=space.dtype)
    low = np.array([0], dtype=space.dtype)

    return ActionSpace(high, low)


def fromBox(
        space: Box
        ):

    high = space.high
    low = space.low

    return ActionSpace(high, low)
```

# 9월 16일

## 1. ./module/utils/ActionSpace/ActionSpace.py __init__.py 수정

```python
        # If it has its own actionSpace
        if (high is None and low is None) and actionSpace is not None:

            if type(actionSpace) == Discrete:
                self.__dict__ = fromDiscrete(actionSpace).__dict__
            if type(actionSpace) == Box:
                self.__dict__ = fromBox(actionSpace).__dict__

            if not (type(actionSpace) in [Discrete, Box]):
                raise ValueError(
                    f"Supported Action Spaces are {str(Discrete)}, {str(Box)}")

        # When high and low given
        elif (high is not None and low is not None) and actionSpace is None:

            self.dtype = high.dtype  # Data type of each element
            self.high = high  # Biggest values of each element
            self.low = low  # Smallest values of each element
            self.shape = high.shape  # Shape of Actions

            # Check Validity
            # int or float
            if self.dtype in [np.int32, np.int64]:
                self.actionType = 'Discrete'
            elif self.dtype in [np.float32, np.float64]:
                self.actionType = 'Continuous'
            else:
                raise ValueError(
                        "Action Space data type should be float or integer")

            # Check data type and shape
            if self.high.dtype != self.low.dtype:
                raise ValueError(
                        "high and low have different data type!")
            if self.high.shape != self.low.shape:
                raise ValueError(
                        "high and low have different shape!")

        else:
            raise ValueError(
                    "")
```

1. high와 low를 받으면 맞춰서 ActionSpace를 생성한다.<br/>
2. 외부의 Action Space를 사용하고 싶으면 actionSpace 파라미터를 사용할 수 있다. (우선 gym의 Box와 Discrete를 지원)<br/>

## 2. ./module/Policy/Policy.py의 ContinuousPolicy class를 작성

```python
class ContinuousPolicy(Policy):

    def __init__(
            self,
            algorithm: str = "plain",  # plain
            exploring: str = "normal",  # normal, None
            exploringParams: Dict[str, Union[int, float]] = {
                'mean': 0,  # mean
                'sigma': 1,  # standard deviation
            },):

        super().__init__(
                algorithm=algorithm,
                exploring=exploring,
                exploringParams=exploringParams,
                )

        # Initialize Parameters
        if exploring == 'normal':
            self.useEps = True
            self.exploration = NormalNoise(
                    **exploringParams)

    # Return Action
    @overrides(Policy)
    def __call__(
            self,
            actionValue: torch.Tensor,
            stepsDone: int,
            ) -> torch.Tensor:

        # Get noise
        noise = self.exploration(
                stepsDone,
                actionValue.shape)

        # Add noise to action
        action = actionValue + noise

        return action
```

## 3. 모양에 맞는 noise 생성을 위한 ./module/utils/exploration.py의 NormalNoise class 수정

```python
class NormalNoise():

    def __init__(
            self,
            mean: float,
            sigma: float):

        self._mu = mean
        self._sigma = sigma

    def __call__(
            self,
            stepsDone: int,
            shape: Union[tuple, int],
            ) -> float:

        return np.random.normal(self._mu, self._sigma, shape)
```

## 4.  Value Function의 역할을 하는 ./module/PG/Value/Value.py 의 Value class 초기화 수정

```python
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer,
        actionSpace: ActionSpace,
        actionParams: Dict[str, Union[int, float, Dict]] = None,
        clippingParams: Dict[str, Union[int, float]] = {
            'pNormValue': 2,
            'maxNorm': 1,
        },
    ):

        # Initialize Parameter
        self.model = model
        self.optimizer = optimizer
        self.clippintParams = clippingParams
        self.actionSpace = actionSpace
        self.stepsDone = 0

        # Set policy
        if self.actionSpace.actionType == 'Discrete':

            # default actionParams
            if actionParams is None:
                actionParams = {
                    'algorithm': "greedy",  # greedy, stochastic
                    'exploring': "epsilon",  # epsilon, None
                    'exploringParams': {
                        'start': 0.99,
                        'end': 0.0001,
                        'decay': 10000
                    }
                }

            self.policy = DiscretePolicy(**actionParams)

        if self.actionSpace.actionType == 'Continuous':

            # default actionParams
            if actionParams is None:
                actionParams = {
                    'algorithm': "plain",  # greedy
                    'exploring': "normal",  # normal
                    'exploringParams': {
                        'mean': 0,
                        'sigma': 1,
                    }
                }

            self.policy = ContinuousPolicy(**actionParams)

        if self.actionSpace.actionType \
                not in ['Discrete', 'Continuous']:

            raise ValueError(
                "actionType only for Discrete and Continuous Action")
```

1. model, optimizer, clippingParams등의 property를 초기화<br/>
2. actionParams을 설정하지 않은 경우 default를 사용하도록 작성<br/><br/>

**이 Value class는 Value Function의 역할을 제외하고도 다음과 같은 역할을 한다.**<br/>
1. Policy property를 가지고 행동을 출력한다.<br/>
2. Value Function의 역할을 하는 뉴럴넷을 property로 가지고 관리한다.<br/>

