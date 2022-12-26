import torch
import numpy as np

if __name__ == "__main__":
    from tetris import *
else:
    from module.envs.Tetris.tetris import *


class TetrisEnv_v0():

    def __init__(self):
        self.board = Board()
        self.board.start()

    # Return next_state, reward, done, action
    def step(
            self,
            action: torch.Tensor) \
            -> tuple[np.ndarray, float, bool, torch.Tensor]:

        # action
        # 0: Move Left
        # 1: Move Right
        # 2: Rotate Left
        # 3: Rotate Right
        action = int(action)  # torch tensor to int
        self.board.move(action)

    def render(self):
        pass

    def reset(self):
        self.tetris.tboard.initBoard()

    def close(self):
        pass


if __name__ == "__main__":
    env = TetrisEnv_v0()
    env.step(1)
