import random
from typing import Tuple
import numpy as np

from PyQt6.QtCore import Qt, QBasicTimer, pyqtSignal
from PyQt6.QtGui import QPainter, QColor
from PyQt6.QtWidgets import QMainWindow, QFrame, QApplication


class Tetris_AI_v2(QMainWindow):

    def __init__(self, rl_object):
        super().__init__()

        self.initUI(rl_object)

    def initUI(self, rl_object):
        """initiates application UI"""

        self.tboard = Board_AI(self, rl_object)
        self.setCentralWidget(self.tboard)

        self.statusbar = self.statusBar()
        self.tboard.msg2Statusbar[str].connect(self.statusbar.showMessage)

        self.tboard.start()

        self.resize(320, 600)
        self.center()
        self.setWindowTitle('Tetris by ComEdu at KNU openscience')
        self.show()

    def center(self):
        """centers the window on the screen"""

        qr = self.frameGeometry()
        cp = self.screen().availableGeometry().center()

        qr.moveCenter(cp)
        self.move(qr.topLeft())


class Board_AI(QFrame):

    msg2Statusbar = pyqtSignal(str)

    BoardWidth = 10
    BoardHeight = 22
    Speed = 300

    oneLineDropPer = 10  # Every 10 action one line drops
    num_shapes = 7

    def __init__(self, parent, rl_object):
        super().__init__(parent)

        self.initBoard()
        
        self.rl_object = rl_object
        self.moveCnt = 0

    def initBoard(self):
        """initiates board"""

        self.timer = QBasicTimer()
        self.isWaitingAfterLine = False

        self.curX = 0
        self.curY = 0
        self.numLinesRemoved = 0
        self.board = []

        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.isStarted = False
        self.isPaused = False
        self.clearBoard()

    def shapeAt(self, x, y):
        """determines shape at the board position"""

        return self.board[(y * Board_AI.BoardWidth) + x]

    def setShapeAt(self, x, y, shape):
        """sets a shape at the board"""

        self.board[(y * Board_AI.BoardWidth) + x] = shape

    def squareWidth(self):
        """returns the width of one square"""

        return self.contentsRect().width() // Board_AI.BoardWidth

    def squareHeight(self):
        """returns the height of one square"""

        return self.contentsRect().height() // Board_AI.BoardHeight

    def start(self):
        """starts game"""

        if self.isPaused:
            return

        self.isStarted = True
        self.isWaitingAfterLine = False
        self.numLinesRemoved = 0
        self.clearBoard()

        self.msg2Statusbar.emit(str(self.numLinesRemoved))

        self.newPiece()
        self.timer.start(Board_AI.Speed, self)

    def pause(self):
        """pauses game"""

        if not self.isStarted:
            return

        self.isPaused = not self.isPaused

        if self.isPaused:
            self.timer.stop()
            self.msg2Statusbar.emit("paused")

        else:
            self.timer.start(Board_AI.Speed, self)
            self.msg2Statusbar.emit(str(self.numLinesRemoved))

        self.update()

    def paintEvent(self, event):
        """paints all shapes of the game"""

        painter = QPainter(self)
        rect = self.contentsRect()

        boardTop = rect.bottom() - Board_AI.BoardHeight * self.squareHeight()

        for i in range(Board_AI.BoardHeight):
            for j in range(Board_AI.BoardWidth):
                shape = self.shapeAt(j, Board_AI.BoardHeight - i - 1)

                if shape != Tetrominoe.NoShape:
                    self.drawSquare(painter,
                                    rect.left() + j * self.squareWidth(),
                                    boardTop + i * self.squareHeight(), shape)

        if self.curPiece.shape() != Tetrominoe.NoShape:

            for i in range(4):
                x = self.curX + self.curPiece.x(i)
                y = self.curY - self.curPiece.y(i)
                self.drawSquare(painter, rect.left() + x * self.squareWidth(),
                            boardTop + (Board_AI.BoardHeight - y - 1) * self.squareHeight(),
                            self.curPiece.shape())

    def keyPressEvent(self, event):
        """processes key press events"""

        if not self.isStarted or self.curPiece.shape() == Tetrominoe.NoShape:
            super(Board_AI, self).keyPressEvent(event)
            return

        key = event.key()

        if key == Qt.Key.Key_P:
            self.pause()
            return

        if self.isPaused:
            return

        elif key == Qt.Key.Key_Left.value:
            self.tryMove(self.curPiece, self.curX - 1, self.curY)

        elif key == Qt.Key.Key_Right.value:
            self.tryMove(self.curPiece, self.curX + 1, self.curY)

        elif key == Qt.Key.Key_Down.value:
            self.tryMove(self.curPiece.rotateRight(), self.curX, self.curY)

        elif key == Qt.Key.Key_Up.value:
            self.tryMove(self.curPiece.rotateLeft(), self.curX, self.curY)

        elif key == Qt.Key.Key_Space.value:
            self.dropDown()

        elif key == Qt.Key.Key_D.value:
            self.oneLineDown()

        else:
            super(Board_AI, self).keyPressEvent(event)

    def timerEvent(self, event):
        """handles timer event"""

        if event.timerId() == self.timer.timerId():

            if self.isWaitingAfterLine:
                self.isWaitingAfterLine = False
                self.newPiece()
            else:
                self.move_AI()
                self.oneLineDown()

        else:
            super(Board_AI, self).timerEvent(event)

    def clearBoard(self):
        """clears shapes from the board"""

        for i in range(Board_AI.BoardHeight * Board_AI.BoardWidth):
            self.board.append(Tetrominoe.NoShape)

    def dropDown(self):
        """drops down a shape"""

        newY = self.curY

        while newY > 0:

            if not self.tryMove(self.curPiece, self.curX, newY - 1):
                break

            newY -= 1

        self.pieceDropped()

    def oneLineDown(self):
        """goes one line down with a shape"""

        if not self.tryMove(self.curPiece, self.curX, self.curY - 1):
            self.pieceDropped()

    def pieceDropped(self):
        """after dropping shape, remove full lines and create new shape"""

        for i in range(4):
            x = self.curX + self.curPiece.x(i)
            y = self.curY - self.curPiece.y(i)
            self.setShapeAt(x, y, self.curPiece.shape())

        self.removeFullLines()

        if not self.isWaitingAfterLine:
            self.newPiece()

    def removeFullLines(self):
        """removes all full lines from the board"""

        numFullLines = 0
        rowsToRemove = []

        for i in range(Board_AI.BoardHeight):

            n = 0
            for j in range(Board_AI.BoardWidth):
                if not self.shapeAt(j, i) == Tetrominoe.NoShape:
                    n = n + 1

            if n == 10:
                rowsToRemove.append(i)

        rowsToRemove.reverse()

        for m in rowsToRemove:

            for k in range(m, Board_AI.BoardHeight):
                for l in range(Board_AI.BoardWidth):
                    self.setShapeAt(l, k, self.shapeAt(l, k + 1))

        numFullLines = numFullLines + len(rowsToRemove)

        if numFullLines > 0:
            self.numLinesRemoved = self.numLinesRemoved + numFullLines
            self.msg2Statusbar.emit(str(self.numLinesRemoved))

            self.isWaitingAfterLine = True
            self.curPiece.setShape(Tetrominoe.NoShape)
            self.update()

    def newPiece(self):
        """creates a new shape"""

        self.curPiece = Shape()
        self.curPiece.setRandomShape()
        self.curX = Board_AI.BoardWidth // 2 + 1
        self.curY = Board_AI.BoardHeight - 1 + self.curPiece.minY()

        if not self.tryMove(self.curPiece, self.curX, self.curY):

            self.curPiece.setShape(Tetrominoe.NoShape)
            self.timer.stop()
            self.isStarted = False
            self.msg2Statusbar.emit("Game over")

    def tryMove(self, newPiece, newX, newY):
        """tries to move a shape"""

        for i in range(4):

            x = newX + newPiece.x(i)
            y = newY - newPiece.y(i)

            if x < 0 or x >= Board_AI.BoardWidth or y < 0 or y >= Board_AI.BoardHeight:
                return False

            if self.shapeAt(x, y) != Tetrominoe.NoShape:
                return False

        self.curPiece = newPiece
        self.curX = newX
        self.curY = newY
        self.update()

        return True

    def drawSquare(self, painter, x, y, shape):
        """draws a square of a shape"""

        colorTable = [0x000000, 0xCC6666, 0x66CC66, 0x6666CC,
                      0xCCCC66, 0xCC66CC, 0x66CCCC, 0xDAAA00]

        color = QColor(colorTable[shape])
        painter.fillRect(x + 1, y + 1, self.squareWidth() - 2,
                         self.squareHeight() - 2, color)

        painter.setPen(color.lighter())
        painter.drawLine(x, y + self.squareHeight() - 1, x, y)
        painter.drawLine(x, y, x + self.squareWidth() - 1, y)

        painter.setPen(color.darker())
        painter.drawLine(x + 1, y + self.squareHeight() - 1,
                         x + self.squareWidth() - 1,
                         y + self.squareHeight() - 1)
        painter.drawLine(x + self.squareWidth() - 1,
                         y + self.squareHeight() - 1,
                         x + self.squareWidth() - 1, y + 1)

    # ===========================================
    # Belows are for Reinforcment Learning system
    # ===========================================

    def get_state(self) -> Tuple[np.ndarray, list]:

        # Check the walls, (check self.board)
        board_state = [1 if i != 0 else 0 for i in self.board]
        two_dim_board = []
        for height in reversed(range(self.BoardHeight)):
            two_dim_board.append(board_state[
                self.BoardWidth*height:
                self.BoardWidth*height + self.BoardWidth])

        # Create controlable block data as 2d list
        block_board = [
                [0] * self.BoardWidth
                for _ in range(self.BoardHeight)]
        coords = self.curPiece.coords
        posX = self.curX
        posY = self.BoardHeight - (self.curY + 1)

        # Fill the positions that block exists
        block_board[posY + coords[0][1]][posX + coords[0][0]] = 1
        block_board[posY + coords[1][1]][posX + coords[1][0]] = 1
        block_board[posY + coords[2][1]][posX + coords[2][0]] = 1
        block_board[posY + coords[3][1]][posX + coords[3][0]] = 1

        # get map information as 3d array
        map_state = [two_dim_board, block_board]
        map_state = np.array(map_state)

        return map_state

    def move(
            self,
            action: int):
        """processes key press events"""

        if action == 0:  # Move Left
            self.tryMove(self.curPiece, self.curX - 1, self.curY)

        elif action == 1:  # Move Right
            self.tryMove(self.curPiece, self.curX + 1, self.curY)

        elif action == 2:  # Rotate Right
            self.tryMove(self.curPiece.rotateRight(), self.curX, self.curY)

        elif action == 3:  # Rotate Left
            self.tryMove(self.curPiece.rotateLeft(), self.curX, self.curY)

        elif action == 4:  # Drop down
            self.dropDown()

        else:
            raise ValueError(
                    "Action is out of bound!!"
                    "only 0, 1, 2, 3 are possible")

        self.moveCnt += 1

    def move_AI(self):
        for _ in range(self.oneLineDropPer):
            state = self.get_state()
            action = self.rl_object.value.get_action(
                    state,
                    isTest=True)
            self.move(action)


class Tetrominoe:

    NoShape = 0
    ZShape = 1
    SShape = 2
    LineShape = 3
    TShape = 4
    SquareShape = 5
    LShape = 6
    MirroredLShape = 7


class Shape:

    coordsTable = (
        ((0, 0), (0, 0), (0, 0), (0, 0)),
        ((0, -1), (0, 0), (-1, 0), (-1, 1)),
        ((0, -1), (0, 0), (1, 0), (1, 1)),
        ((0, -1), (0, 0), (0, 1), (0, 2)),
        ((-1, 0), (0, 0), (1, 0), (0, 1)),
        ((0, 0), (1, 0), (0, 1), (1, 1)),
        ((-1, -1), (0, -1), (0, 0), (0, 1)),
        ((1, -1), (0, -1), (0, 0), (0, 1))
    )

    def __init__(self):

        self.coords = [[0, 0] for i in range(4)]
        self.pieceShape = Tetrominoe.NoShape

        self.setShape(Tetrominoe.NoShape)

    def shape(self):
        """returns shape"""

        return self.pieceShape

    def setShape(self, shape):
        """sets a shape"""

        table = Shape.coordsTable[shape]

        for i in range(4):
            for j in range(2):
                self.coords[i][j] = table[i][j]

        self.pieceShape = shape

    def setRandomShape(self):
        """chooses a random shape"""

        self.setShape(random.randint(1, 7))

    def x(self, index):
        """returns x coordinate"""

        return self.coords[index][0]

    def y(self, index):
        """returns y coordinate"""

        return self.coords[index][1]

    def setX(self, index, x):
        """sets x coordinate"""

        self.coords[index][0] = x

    def setY(self, index, y):
        """sets y coordinate"""

        self.coords[index][1] = y

    def minX(self):
        """returns min x value"""

        m = self.coords[0][0]
        for i in range(4):
            m = min(m, self.coords[i][0])

        return m

    def maxX(self):
        """returns max x value"""

        m = self.coords[0][0]
        for i in range(4):
            m = max(m, self.coords[i][0])

        return m

    def minY(self):
        """returns min y value"""

        m = self.coords[0][1]
        for i in range(4):
            m = min(m, self.coords[i][1])

        return m

    def maxY(self):
        """returns max y value"""

        m = self.coords[0][1]
        for i in range(4):
            m = max(m, self.coords[i][1])

        return m

    def rotateLeft(self):
        """rotates shape to the left"""

        if self.pieceShape == Tetrominoe.SquareShape:
            return self

        result = Shape()
        result.pieceShape = self.pieceShape

        for i in range(4):
            result.setX(i, self.y(i))
            result.setY(i, -self.x(i))

        return result

    def rotateRight(self):
        """rotates shape to the right"""

        if self.pieceShape == Tetrominoe.SquareShape:
            return self

        result = Shape()
        result.pieceShape = self.pieceShape

        for i in range(4):
            result.setX(i, -self.y(i))
            result.setY(i, self.x(i))

        return result


if __name__ == "__main__":

    import sys
    sys.path.append("../../../")

    import torch
    import torch.optim as optim
    from grll.PG.models import CNN_V4
    from grll.PG import A2C
    from grll.envs.Tetris import TetrisEnv_v2

    # device to use
    device = torch.device('cpu')

    # set environment
    env = TetrisEnv_v2()

    num_actions = env.num_actions
    num_states = env.num_obs
    ALPHA = 1e-4  # learning rate

    model = CNN_V4(num_states, num_actions).to(device)
    optimizer = optim.SGD(model.parameters(), lr=ALPHA, momentum=0.9)

    Agent = A2C(
        device=device,  # device to use, 'cuda' or 'cpu'
        env=env,
        model=model,  # torch models for policy and value funciton
        optimizer=optimizer,  # torch optimizer
    )

    Agent.load(
            "../../../TRIALS/saved_models/TetrisEnv_v2/" +
            "A2C_CNN_V4_SGD_lr1e-4_step_30000000.obj")

    # Below codes are not working
    app = QApplication([])
    tetris = Tetris_AI_v2(Agent)
    app.exec()

    print(tetris.tboard.moveCnt)
