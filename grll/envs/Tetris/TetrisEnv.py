import torch
import numpy as np
from PyQt6.QtCore import QBasicTimer

if __name__=="__main__":
    from tetris import *
else:
    from module.envs.Tetris.tetris import *


class TetrisEnv_v0():
    
    def __init__(self):
        
        self.app = QApplication([])
        self.tetris = Tetris()
        self.app.exec()

        self.step(1)

    # Return next_state, reward, done, action
    def step(
            self,
            action: torch.Tensor) \
            -> tuple[np.ndarray, float, bool, torch.Tensor]:

        import time
        time.sleep(10)

        self.tetris.tboard.pause()

    def render(self):
        pass

    def reset(self):
        self.tetris.tboard.initBoard()

    def close(self):
        pass


if __name__ == "__main__":
    env = TetrisEnv_v0()
