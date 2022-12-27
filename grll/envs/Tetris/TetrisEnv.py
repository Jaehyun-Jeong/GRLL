import torch
import numpy as np

if __name__ == "__main__":
    from tetris import *
else:
    from module.envs.Tetris.tetris import *


class TetrisEnv_v0():

    num_shapes = 7

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
        # 4: Drop down
        action = int(action)  # torch tensor to int
        self.board.move(action)

    def get_state(self):

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

        coords = self.board.curPiece.coords
        posX = self.board.curX - 1
        posY = self.board.BoardHeight - (self.board.curY + 1)
        two_dim_board[posY + coords[0][1]][posX + coords[0][0]] = 2
        two_dim_board[posY + coords[1][1]][posX + coords[1][0]] = 2
        two_dim_board[posY + coords[2][1]][posX + coords[2][0]] = 2
        two_dim_board[posY + coords[3][1]][posX + coords[3][0]] = 2

        for line in two_dim_board:
            print(line)
        print("=====================================")

        # piece shape number to one hot encoding
        shape_onehot = [0 for _ in range(TetrisEnv_v0.num_shapes)]
        shape_onehot[self.board.curPiece.shape()-1] = 1

        '''
        print(f"X: {self.board.curX}")
        print(f"Y: {self.board.curY}")
        print(block_height)
        print(f"return: {self.board.numLinesRemoved}")
        print(shape_onehot)
        print(self.board.curPiece.coords)
        print(self.board.curPiece.pieceShape)
        '''

    def render(self):
        pass

    def reset(self):
        self.tetris.tboard.initBoard()

    def close(self):
        pass


if __name__ == "__main__":
    env = TetrisEnv_v0()
    action_list = [4, 1, 3, 1, 1, 3]
    # action_list = [4, 1, 1, 1, 4, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 4, 0, 0, 4, 1, 4]

    for action in action_list:
        env.step(action)
        env.get_state()

