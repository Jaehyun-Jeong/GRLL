from typing import Union, Tuple, List

import sys
sys.path.append("../../../")

from grll.envs.Maze.Maze_Generator import Maze
from grll.utils.ActionSpace.ActionSpace import ActionSpace

from copy import deepcopy
import torch
from random import choice
import numpy as np
import pygame


class MazeEnv_base():

    BLOCK_SIZE = 20

    def __init__(
            self,
            mazeSize: Tuple[int, int] = (9, 9),
            maze: np.array = None,  # Custom Maze, If it is None than create random Maze
        ):

        # If mazeSize and size of maze not matching
        if maze and not mazeSize == maze.shape:
            raise Value

        self.mazeSize = mazeSize

        # Set display size
        self.displaySize = (
                self.mazeSize[0]*self.BLOCK_SIZE,
                self.mazeSize[1]*self.BLOCK_SIZE)

        # Create new maze and set player and goal
        self.initNewMaze()

        # When you have display set pygame screen as pygame.display
        # But, if you not set it as pygame.Surface which don't have to be displayed
        try:
            self.screen = pygame.display.set_mode(
                    self.displaySize, flags=pygame.HIDDEN)
        except pygame.error:
            self.screen = pygame.Surface(self.displaySize)

        self.isRender = False

    def initNewMaze(self):
        # set blocks and character = 2, goal = 3
        self.blocks = self.makeMaze()
        self.blocks[self.blocks.shape[0]-2][self.blocks.shape[1]-2] = 2
        self.blocks[1][1] = 3

    # Only use when displaying environment
    def displaySetting(self):

        # =================================================================================
        # IMG PATH SETTING
        # =================================================================================

        import os.path as path
        root_dir = path.join(path.dirname(path.abspath(__file__)), 'imgs')

        self.ROAD_IMG = path.join(root_dir, "road.jpg")
        self.CHARACTER_IMG = path.join(root_dir, "character.jpg")
        self.GOAL_IMG = path.join(root_dir, "goal.jpg")

        self.wallName_to_Info = {
            # it was made to pick woods img by index
            # east, west, south, north wall information
            # 1: no wall linked, 2: wall linked
            path.join(root_dir, "wall_01.jpg"): [1, 2, 1, 1],
            path.join(root_dir, "wall_02.jpg"): [2, 1, 1, 1],
            path.join(root_dir, "wall_03.jpg"): [1, 1, 2, 1],
            path.join(root_dir, "wall_04.jpg"): [1, 1, 1, 2],
            path.join(root_dir, "wall_05.jpg"): [1, 2, 2, 1],
            path.join(root_dir, "wall_06.jpg"): [2, 1, 2, 1],
            path.join(root_dir, "wall_07.jpg"): [1, 2, 1, 2],
            path.join(root_dir, "wall_08.jpg"): [2, 1, 1, 2],
            path.join(root_dir, "wall_09.jpg"): [1, 1, 2, 2],
            path.join(root_dir, "wall_10.jpg"): [2, 2, 1, 2],
            path.join(root_dir, "wall_11.jpg"): [2, 2, 2, 1],
            path.join(root_dir, "wall_12.jpg"): [2, 1, 2, 2],
            path.join(root_dir, "wall_13.jpg"): [1, 2, 2, 2],
            path.join(root_dir, "wall_14.jpg"): [2, 2, 2, 2],
            path.join(root_dir, "wall_15.jpg"): [2, 2, 1, 1],
        }

        # =================================================================================

        self.loadImages()
        self.imgMat = self.__init_ImgMat()

    def makeMaze(
            self,
            maze: np.ndarray = None):

        # set screen size and initialize it
        disp_size = self.displaySize
        # the rect inside which to draw the maze. Top x, top y, width, height.
        rect = np.array([0, 0, disp_size[0], disp_size[1]])
        screen = pygame.Surface(disp_size)
        pygame.display.set_caption('Maze')

        block_size = 20  # block size in pixels

        # intialize a maze, given size (y, x)
        self.maze = Maze(rect[2] // (block_size * 2) - 1, rect[3] // (block_size * 2) - 1)
        # if this is set, the maze generation process will be displayed in a window. otherwise not.
        self.maze.screen = screen
        screen.fill((0, 0, 0))
        self.maze.screen_block_size = np.min(rect[2:4] / np.flip(self.maze.block_size))
        self.maze.screen_block_offset = rect[0:2] + (rect[2:4] - self.maze.screen_block_size * np.flip(self.maze.block_size)) // 2
        
        if maze:  # If you using Custom Maze
            self.maze.blocks = maze
        else:
            self.maze.gen_maze_2D()

        return self.maze.blocks

    def loadImages(self):
        
        self.imgs = {}
        wallDir = list(self.wallName_to_Info.keys())
        otherDir = [self.ROAD_IMG, self.CHARACTER_IMG, self.GOAL_IMG]
        
        for imgFileName in wallDir + otherDir:
            img = pygame.image.load(imgFileName)
            img = pygame.transform.scale(img, size=((self.maze.screen_block_size,)*2))
            self.imgs[imgFileName] = img

    def __init_ImgMat(self):

        imgInfoMat = np.zeros(self.maze.blocks.shape+(4,))
        imgMat = np.zeros(self.maze.blocks.shape, dtype=object)
        notWallIdx = [0, 2, 3]  # in self.blocks 0 is road 2 is character, and 3 is goal
        
        for row in range(imgInfoMat.shape[0]):
            for col in range(imgInfoMat.shape[1]):
                wallInfo = [0, 0, 0, 0]
                if self.maze.blocks[row][col] == 1: # if its wall
                    
                    if col+1 < imgInfoMat.shape[1]:
                        wallInfo[0] = 1 if self.maze.blocks[row][col+1] in notWallIdx else 2
                    else:
                        wallInfo[0] = 1
                    
                    if col > 0:
                        wallInfo[1] = 1 if self.maze.blocks[row][col-1] in notWallIdx else 2
                    else:
                        wallInfo[1] = 1
                            
                    if row+1 < imgInfoMat.shape[0]:
                        wallInfo[2] = 1 if self.maze.blocks[row+1][col] in notWallIdx else 2
                    else:
                        wallInfo[2] = 1

                    if row > 0:
                        wallInfo[3] = 1 if self.maze.blocks[row-1][col] in notWallIdx else 2
                    else:
                        wallInfo[3] = 1

                    wallName = self.wallInfoToName(wallInfo)
                    imgMat[row][col] = wallName
                    imgInfoMat[row][col] = self.wallName_to_Info[wallName]

                elif self.maze.blocks[row][col] == 0: # if its road
                    imgMat[row][col] = self.ROAD_IMG
                elif self.maze.blocks[row][col] == 2: # if its character
                    imgMat[row][col] = self.CHARACTER_IMG
                elif self.maze.blocks[row][col] == 3: # if its goal
                    imgMat[row][col] = self.GOAL_IMG

        return imgMat
    
    def wallInfoToName(self, wallInfoLst: list): # wallInfoLst like [2, 1, 0, 3] from makeCellImgMat method
    
        possibleWalls = self.wallName_to_Info.copy()
        
        for wallName, wallInfo in self.wallName_to_Info.items():
            
            if ((wallInfoLst[0] != 0 and wallInfoLst[0] != wallInfo[0]) or
               (wallInfoLst[1] != 0 and wallInfoLst[1] != wallInfo[1]) or
               (wallInfoLst[2] != 0 and wallInfoLst[2] != wallInfo[2]) or
               (wallInfoLst[3] != 0 and wallInfoLst[3] != wallInfo[3])):
                
                possibleWalls.pop(wallName)
            
        return choice(list(possibleWalls.keys()))

    def init_draw(self):
        for row in range(self.imgMat.shape[0]):
            for col in range(self.imgMat.shape[1]):
                img = self.imgs[self.imgMat[row][col]]
                imgPos = (col * img.get_width(), row * img.get_height())
                self.screen.blit(img, imgPos)

    def draw(self):
        # get Road image and position
        roadImg = self.imgs[self.ROAD_IMG]
        prevCharPos = np.where(self.imgMat == self.CHARACTER_IMG)
        prevCharPos = (prevCharPos[0][0], prevCharPos[1][0])
        self.imgMat[prevCharPos[0], prevCharPos[1]] = self.ROAD_IMG
        prevImgPos = (
                prevCharPos[1] * roadImg.get_width(),
                prevCharPos[0] * roadImg.get_height())

        # get Character image and position
        charImg = self.imgs[self.CHARACTER_IMG]
        charPos = self.get_char_pos()
        self.imgMat[charPos[0], charPos[1]] = self.CHARACTER_IMG
        imgPos = (
                charPos[1] * charImg.get_width(),
                charPos[0] * charImg.get_height())

        # show images in screen
        self.screen.blit(roadImg, prevImgPos)
        self.screen.blit(charImg, imgPos)

    def render(self):
        try:
            if not self.isRender:
                pygame.display.set_mode(self.displaySize, flags=pygame.SHOWN)
                self.isRender = True
                self.init_draw()
            self.draw()
            pygame.display.update()
        except:
            raise RuntimeError("No available display to render")

    def get_char_pos(self) -> tuple:
        pos = np.where(self.blocks == 2)
        pos = (pos[0][0], pos[1][0])

        return pos


class MazeEnv_v0(MazeEnv_base):

    characterValue = 0.5
    goalValue = 0.3

    def __init__(self):
        super().__init__()

        # Left, Right, Up, Down
        self.num_action = 4
        self.num_obs = self.blocks.shape[0] * self.blocks.shape[1]

        self.action_space = ActionSpace(
                high=np.array([3]),
                low=np.array([0]))

    def reset(self) -> np.ndarray:

        # To make render method work, It should be initialized as False
        self.isRender = False

        charPos = self.get_char_pos()
        self.blocks[charPos[0]][charPos[1]] = 0
        self.blocks[self.blocks.shape[0]-2][self.blocks.shape[1]-2] = 2
        self.blocks[1][1] = 3

        if pygame.display.get_active():
            pygame.display.set_mode(self.displaySize, flags=pygame.HIDDEN)

        return self.get_state()

    def step(self, action: Union[int, torch.Tensor]) \
            -> Tuple[np.ndarray, float, bool, torch.Tensor]:

        # Get reward
        _, done = self.move(action)
        if done:
            reward = 1
        else:
            reward = -0.04

        next_state = self.get_state()

        return next_state, reward, done, action

    def get_state(self):
        state = self.blocks.flatten()
        state = state.astype(np.float32)

        # Find Character index and change the value
        charPos = self.get_char_pos()
        charPosIdx = charPos[0]*self.blocks.shape[0] + charPos[1]
        state[charPosIdx] = self.characterValue

        state[self.blocks.shape[0]+1] = self.goalValue

        return state

    def move(
            self,action: Union[int, torch.Tensor, np.ndarray]) -> bool:
        
        if action not in [0, 1, 2, 3]:
            raise ValueError("Action is out of bound")

        # get Position to move
        charPos = self.get_char_pos()
        movePos = list(deepcopy(charPos))
        if action == 0:  # east
            movePos[1] += 1
        if action == 1:  # west
            movePos[1] -= 1
        if action == 2:  # south
            movePos[0] += 1
        if action == 3:  # north
            movePos[0] -= 1
        movePos = tuple(movePos)

        if self.blocks[movePos[0]][movePos[1]] == 0:
            self.blocks[movePos[0]][movePos[1]] = 2  # Road Pos to Char
            self.blocks[charPos[0]][charPos[1]] = 0  # Char Pos to Raod
            moved = True
            done = False
        elif self.blocks[movePos[0]][movePos[1]] == 3:  # Value 3 means goal
            self.blocks[movePos[0]][movePos[1]] = 2  # Road Pos to Char
            self.blocks[charPos[0]][charPos[1]] = 0  # Char Pos to Raod
            moved = True
            done = True
        else:  # not movable
            moved = False
            done = False

        return moved, done

    def close(self):
        pygame.quit()


# This one added return information like below
# 1. Track the character moved and if character reach that spot, then give -.25 reward
# 2. If it hits the wall then give -.75 reward
class MazeEnv_v1(MazeEnv_v0):

    passedValue = 4  # To indicate road that character passed

    def __init__(self):

        super().__init__()

    def step(self, action: Union[int, torch.Tensor]) \
            -> Tuple[np.ndarray, float, bool, torch.Tensor]:

        # Get reward
        moved, done, passed = self.move(action)
        hitWall = not moved
        if done:
            reward = 1
        elif hitWall:
            reward = -0.75
        elif passed:
            reward = -0.25
        else:
            reward = -0.04

        next_state = self.get_state()

        return next_state, reward, done, action

    def get_state(self):
        state = super().get_state()
        state[state == self.passedValue] = 0.7

        return state

    def move(
            self,action: Union[int, torch.Tensor, np.ndarray]) -> bool:
        
        if action not in [0, 1, 2, 3]:
            raise ValueError("Action is out of bound")

        # get Position to move
        charPos = self.get_char_pos()
        movePos = list(deepcopy(charPos))
        if action == 0:  # east
            movePos[1] += 1
        if action == 1:  # west
            movePos[1] -= 1
        if action == 2:  # south
            movePos[0] += 1
        if action == 3:  # north
            movePos[0] -= 1
        movePos = tuple(movePos)

        if self.blocks[movePos[0]][movePos[1]] == 0:
            self.blocks[movePos[0]][movePos[1]] = 2  # Road Pos to Char
            self.blocks[charPos[0]][charPos[1]] = self.passedValue  # Char Pos to Passed Road
            moved = True
            done = False
            passed = False
        elif self.blocks[movePos[0]][movePos[1]] == 3:  # Value 3 means goal
            self.blocks[movePos[0]][movePos[1]] = 2  # Road Pos to Char
            self.blocks[charPos[0]][charPos[1]] = self.passedValue  # Char Pos to Passed Road
            moved = True
            done = True
            passed = False
        elif self.blocks[movePos[0]][movePos[1]] == 4:  # Value 4 means passed road
            self.blocks[movePos[0]][movePos[1]] = 2  # Road Pos to Char
            self.blocks[charPos[0]][charPos[1]] = self.passedValue  # Char Pos to Passed Road
            moved = True
            done = False
            passed = True

        else:  # not movable
            moved = False
            done = False
            passed = True

        return moved, done, passed


# Added below features
# min_reward: minimum reward agent can reach
# If agent blocked then get "min_reward - 1" which is smaller then min_reward
# Wall as 0 and Road as 1. (It was opposite)
# Support Exploring Starts
class MazeEnv_v2(MazeEnv_v1):

    def __init__(
            self,
            exploring_starts: bool):

        super().__init__()

        # min_reward depend on the maze size
        self.min_reward = -0.5 * \
                self.blocks.shape[0] * self.blocks.shape[1]

        # Cumulative Reward to compare with min_reward
        self.cumulative_reward = 0

        # Exploring Starts
        self.exploring_starts = exploring_starts

    # Opposite wall and road
    def get_state(self):
        state = super().get_state()

        # 0 were roads, 1 were walls
        state[state == 0] = -1  # Temporarily, set walls to -1
        state[state == 0.5] = -2
        state[state > 0] = 0
        state[state == -1] = 1
        state[state == -2] = self.characterValue

        return state

    def step(self, action: Union[int, torch.Tensor]) \
            -> Tuple[np.ndarray, float, bool, torch.Tensor]:

        # Get reward
        moved, done, passed = self.move(action)
        hitWall = not moved
        if done:
            reward = 1
        elif hitWall:
            reward = -0.75
        elif passed:
            reward = -0.25
        else:
            reward = -0.04

        next_state = self.get_state()

        # If cumulative reward is smaller than min_reward, then end the game
        self.cumulative_reward += reward
        done = True if self.cumulative_reward < self.min_reward else done

        return next_state, reward, done, action

    def reset(self) -> np.ndarray:

        self.cumulative_reward = 0

        # Reset passed road
        self.blocks[self.blocks == self.passedValue] = 0

        # To make render method work, It should be initialized as False
        self.isRender = False

        # Delete Character from map
        charPos = self.get_char_pos()
        self.blocks[charPos[0]][charPos[1]] = 0

        # Set goal
        self.blocks[1][1] = 3

        # Exploring starts
        # which means, start at random state
        if self.exploring_starts:
            possible_char_pos = np.transpose(np.where(self.blocks == 0))
            randCharPos = possible_char_pos[np.random.randint(possible_char_pos.shape[0])]
            self.blocks[randCharPos[0]][randCharPos[1]] = 2
        else:
            self.blocks[self.blocks.shape[0]-2][self.blocks.shape[1]-2] = 2

        if pygame.display.get_active():
            pygame.display.set_mode(self.displaySize, flags=pygame.HIDDEN)

        return self.get_state()



if __name__ == "__main__":

    from random import choice 

    env = MazeEnv_v2(exploring_starts=True)
    running = True

    while True:

        state = env.reset()
        done = False

        while not done:
            env.render()
            action = choice([0, 1, 2, 3])
            state, reward, done, _= env.step(action)

            print(env.blocks)
            print(reward)
            print(done)
            input()
