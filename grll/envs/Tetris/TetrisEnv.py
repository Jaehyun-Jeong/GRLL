from typing import Tuple
import torch
import numpy as np

import sys
sys.path.append("../../../")

from grll.envs.Tetris.tetris import Board
from grll.utils.ActionSpace.ActionSpace import ActionSpace


class TetrisEnv_v0():

    num_shapes = 7

    def __init__(self):
        self.board = Board()
        self.board.start()

        # Actions are possible from [0] to [4]
        # 0: Move Left
        # 1: Move Right
        # 2: Rotate Left
        # 3: Rotate Right
        # 4: Drop down
        self.action_space = ActionSpace(
                high=np.array([4]),
                low=np.array([0]))
        self.num_actions = 5
        self.num_obs = 457

    # Return next_state, reward, done, action
    def step(
            self,
            action: torch.Tensor) \
            -> tuple[np.ndarray, float, bool, torch.Tensor]:

        before_return = self.board.numLinesRemoved

        # Actions
        # 0: Move Left
        # 1: Move Right
        # 2: Rotate Left
        # 3: Rotate Right
        # 4: Drop down
        action = int(action)  # torch tensor to int
        self.board.move(action)

        after_return = self.board.numLinesRemoved

        # reward of one step is the removed lines after action
        reward = after_return - before_return
        done = not self.board.isStarted
        next_state = self.get_state()

        return next_state, reward, done, action

    def get_state(self) -> Tuple[np.ndarray, list]:

        # Check the walls, (check self.board.board)
        board_state = [1 if i != 0 else 0 for i in self.board.board]
        two_dim_board = []
        block_height = [0 for _ in range(self.board.BoardWidth)]
        for height in reversed(range(self.board.BoardHeight)):
            two_dim_board.append(board_state[
                self.board.BoardWidth*height:
                self.board.BoardWidth*height + self.board.BoardWidth])

            # If this line has blocks
            # Calculate height of columns
            if 1 in two_dim_board[-1]:
                block_height = [
                        height + 1
                        if (block_height[i] == 0 and v == 1)
                        else block_height[i]
                        for i, v in enumerate(two_dim_board[-1])]

        # Create controlable block data as 2d list
        block_board = [
                [0] * self.board.BoardWidth
                for _ in range(self.board.BoardHeight)]
        coords = self.board.curPiece.coords
        posX = self.board.curX
        posY = self.board.BoardHeight - (self.board.curY + 1)

        # Fill the positions that block exists
        block_board[posY + coords[0][1]][posX + coords[0][0]] = 1
        block_board[posY + coords[1][1]][posX + coords[1][0]] = 1
        block_board[posY + coords[2][1]][posX + coords[2][0]] = 1
        block_board[posY + coords[3][1]][posX + coords[3][0]] = 1

        # piece shape number to one hot encoding
        shape_onehot = [0 for _ in range(TetrisEnv_v0.num_shapes)]
        shape_onehot[self.board.curPiece.shape()-1] = 1

        # get map information as 3d array
        map_state = [two_dim_board, block_board]
        map_state = np.array(map_state)

        # get final state
        state = (map_state, block_height + shape_onehot)

        return state

    def render(self):
        raise NotImplementedError("Not supporting render option")

    def reset(self):

        # Re initialize the board
        self.board.initBoard()
        self.board.start()

        # Return the current state
        return self.get_state()

    def close(self):
        self.reset()


# Return flattened state
class TetrisEnv_v1(TetrisEnv_v0):

    def __init__(self):
        super().__init__()

    # Get flattened state
    def get_state(self):
        state = super().get_state()

        # Flatten and concatenate all states
        flattened_board_state = state[0].flatten()
        rest_state = np.array(state[1])
        state = np.concatenate((flattened_board_state,
                                rest_state))

        return state


# Return flattened state
class TetrisEnv_v2(TetrisEnv_v0):

    def __init__(self):
        super().__init__()
        self.num_obs = (
                2,  # channel: two_dim_board, block_board
                self.board.BoardHeight,
                self.board.BoardWidth)

    # Get flattened state
    def get_state(self) -> np.ndarray:

        # Check the walls, (check self.board.board)
        board_state = [1 if i != 0 else 0 for i in self.board.board]
        two_dim_board = []
        for height in reversed(range(self.board.BoardHeight)):
            two_dim_board.append(board_state[
                self.board.BoardWidth*height:
                self.board.BoardWidth*height + self.board.BoardWidth])

        # Create controlable block data as 2d list
        block_board = [
                [0] * self.board.BoardWidth
                for _ in range(self.board.BoardHeight)]
        coords = self.board.curPiece.coords
        posX = self.board.curX
        posY = self.board.BoardHeight - (self.board.curY + 1)

        # Fill the positions that block exists
        block_board[posY + coords[0][1]][posX + coords[0][0]] = 1
        block_board[posY + coords[1][1]][posX + coords[1][0]] = 1
        block_board[posY + coords[2][1]][posX + coords[2][0]] = 1
        block_board[posY + coords[3][1]][posX + coords[3][0]] = 1

        # get map information as 3d array
        map_state = [two_dim_board, block_board]
        map_state = np.array(map_state)

        return map_state


if __name__ == "__main__":
    env = TetrisEnv_v0()

    # action_list = [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]
    action_list = [4, 1, 1, 1, 4, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 4, 0, 0, 4, 1, 4]

    for action in action_list:
        state, _, _, _ = env.step(action)
        print(state)

    env.reset()
