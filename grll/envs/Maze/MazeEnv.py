from typing import Union, Tuple

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

    def __init__(
            self,
            displaySize: Tuple[int, int] = (500, 500)
        ):

        # set display size
        self.displaySize = displaySize

        # set blocks and character = 2, goal = 3
        self.blocks = self.makeMaze()
        self.blocks[self.blocks.shape[0]-2][self.blocks.shape[1]-2] = 2
        self.blocks[1][1] = 3

        try:
            self.screen = pygame.display.set_mode(
                    self.displaySize, flags=pygame.HIDDEN)
        except pygame.error:
            self.screen = pygame.Surface(self.displaySize)

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
        self.isRender = False

    def makeMaze(self):

        # set screen size and initialize it
        disp_size = self.displaySize
        # the rect inside which to draw the maze. Top x, top y, width, height.
        rect = np.array([0, 0, disp_size[0], disp_size[1]])
        block_size = 20  # block size in pixels
        screen = pygame.Surface(disp_size)
        pygame.display.set_caption('Maze')

        # intialize a maze, given size (y, x)
        self.maze = Maze(rect[2] // (block_size * 2) - 1, rect[3] // (block_size * 2) - 1)
        # if this is set, the maze generation process will be displayed in a window. otherwise not.
        self.maze.screen = screen
        screen.fill((0, 0, 0))
        self.maze.screen_block_size = np.min(rect[2:4] / np.flip(self.maze.block_size))
        self.maze.screen_block_offset = rect[0:2] + (rect[2:4] - self.maze.screen_block_size * np.flip(self.maze.block_size)) // 2

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

    def __init__(self):
        super().__init__()

        # Left, Right, Up, Down
        self.num_action = 4
        self.num_obs = self.blocks.shape[0] * self.blocks.shape[1]

        self.action_space = ActionSpace(
                high=np.array([3]),
                low=np.array([0]))

    def reset(self) -> np.ndarray:

        charPos = self.get_char_pos()
        self.blocks[charPos[0]][charPos[1]] = 0
        self.blocks[self.blocks.shape[0]-2][self.blocks.shape[1]-2] = 2
        self.blocks[1][1] = 3

        if pygame.display.get_active():
            pygame.display.set_mode(self.displaySize, flags=pygame.HIDDEN)

        return self.blocks.flatten()

    def step(self, action: Union[int, torch.Tensor]) \
            -> Tuple[np.ndarray, float, bool, torch.Tensor]:

        # Get reward
        moved, done = self.move(action)
        hitWall = not moved
        if done:
            reward = 1
        else:
            reward = -0.75 if hitWall else -0.04

        next_state = self.blocks.flatten()

        return next_state, reward, done, action

    def move(self, action: Union[int, torch.Tensor]) -> bool:

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


if __name__ == "__main__":

    from random import choice 

    env = MazeEnv_v0()
    running = True

    while True:
        env.render()
        action = choice([0, 1, 2, 3])
        results = env.step(action)

        print("=================================")
        print(type(results[0]))
        print(len(results[0]))
