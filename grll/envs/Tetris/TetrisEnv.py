import torch
import numpy as np

if __name__=="__main__":
    from tetris import *
else:
    from module.envs.Tetris.tetris import *

class TetrisEnv_v0():
    
    def __init__(self):
        
        self.app = QApplication([])
        self.tetris = Tetris()

    # Return next_state, reward, done, action
    def step(
            self,
            action: torch.Tensor) \
            -> tuple[np.ndarray, float, bool, torch.Tensor]

        pass

    def render(self):
        pass

    def reset(self):
        pass

    def close(self):
        pass


if __name__ == "__main__":
    pass
