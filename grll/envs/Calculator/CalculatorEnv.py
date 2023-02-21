from typing import Tuple
import random
import numpy as np
import operator
import torch

import sys
sys.path.append("../../../")

from grll.utils.ActionSpace.ActionSpace import ActionSpace


class CalculatorEnv_v0():

    # It's because smallest number is -9
    INDEXING_GAP = 9
    OPERATOR = {
            '+': operator.add,
            '-': operator.sub,
            '*': operator.mul,
            }
    OPERATOR_LIST = list(OPERATOR.keys())

    # This Environment never going to be end
    DONE = True

    def __init__(self):

        self.num_actions = 110  # From -9 to 100
        # OBS LIST
        # 1. first_number
        # 2. second_number
        # 3. add
        # 4. sub
        # 5. mul
        self.num_obs = 5

        self.action_space = ActionSpace(
                high=np.array([110]),
                low=np.array([0]))

        # To get reward, we should remember previous answer
        self.prev_answer = None

    # Return next_state, reward, done, action
    def step(self, action: torch.Tensor) \
            -> Tuple[np.ndarray, float, bool, torch.Tensor]:

        action = int(action)
        next_state, answer = self.get_state()
        reward = 1 if self.prev_answer + self.INDEXING_GAP == action else 0
        self.prev_answer = answer

        return next_state, reward, self.DONE, action

    def reset(self) -> np.ndarray:

        new_state, answer = self.get_state()
        self.prev_answer = answer

        return new_state

    def close(self):
        self.prev_answer = None

    def render(self):
        raise NotImplementedError(
                "This Environment isn't supporting rendering option!")

    # Return random state and the answer
    def get_state(self) -> np.ndarray:

        first_number = random.randint(1, 10)
        second_number = random.randint(1, 10)

        features = (
                first_number,
                second_number,
                random.choice(self.OPERATOR_LIST),
                )

        answer = self.OPERATOR[features[2]](first_number, second_number)

        # operator to onehot represectation
        onehot = tuple(
                [1 if i == features[2] else 0 for i in self.OPERATOR_LIST])
        features = features[:2] + onehot

        return np.array(features), answer


if __name__ == "__main__":
    cal_env = CalculatorEnv_v0()
    print(cal_env.reset())
    for i in range(10):
        print("=========================")
        A = int(input())
        print(cal_env.step(A))

    cal_env.close()
