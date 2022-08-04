try:
    from module.envs.Maze.Maze_Generator import *
except:
    from Maze_Generator import *
from random import choice

class MazeEnv_v0():
    def __init__(self):
        
        # set blocks and character = 2, goal = 3
        self.blocks = self.makeMaze()
        self.blocks[self.blocks.shape[0]-2][self.blocks.shape[1]-2] = 2
        self.blocks[1][1] = 3
        self.screen = pygame.display.set_mode((self.maze.screen.get_width(), self.maze.screen.get_height()))

        #=================================================================================
        # IMG PATH SETTING
        #=================================================================================

        import os.path as path
        root_dir = path.join(path.dirname(path.abspath(__file__)), 'imgs')
        

        self.ROAD_IMG = path.join(root_dir, "road.jpg")
        self.CHARACTER_IMG = path.join(root_dir, "character.jpg")
        self.GOAL_IMG = path.join(root_dir, "goal.jpg")

        self.wallName_to_Info = {
            # it was made to pick woods img by index
            # east, west, south, north wall information
            # 1: no wall linked, 2: wall linked 
            path.join(root_dir,"wall_01.jpg"): [1, 2, 1, 1], 
            path.join(root_dir,"wall_02.jpg"): [2, 1, 1, 1],
            path.join(root_dir,"wall_03.jpg"): [1, 1, 2, 1],
            path.join(root_dir,"wall_04.jpg"): [1, 1, 1, 2],
            path.join(root_dir,"wall_05.jpg"): [1, 2, 2, 1],
            path.join(root_dir,"wall_06.jpg"): [2, 1, 2, 1],
            path.join(root_dir,"wall_07.jpg"): [1, 2, 1, 2],
            path.join(root_dir,"wall_08.jpg"): [2, 1, 1, 2],
            path.join(root_dir,"wall_09.jpg"): [1, 1, 2, 2],
            path.join(root_dir,"wall_10.jpg"): [2, 2, 1, 2],
            path.join(root_dir,"wall_11.jpg"): [2, 2, 2, 1],
            path.join(root_dir,"wall_12.jpg"): [2, 1, 2, 2],
            path.join(root_dir,"wall_13.jpg"): [1, 2, 2, 2],
            path.join(root_dir,"wall_14.jpg"): [2, 2, 2, 2],
            path.join(root_dir,"wall_15.jpg"): [2, 2, 1, 1],
        }

        #=================================================================================

        self.loadImages()
        self.imgMat = self.makeCellImgMat()
        self.draw()

    def makeMaze(self):

        # set screen size and initialize it
        disp_size = (700, 500)
        rect = np.array([0, 0, disp_size[0], disp_size[1]])  # the rect inside which to draw the maze. Top x, top y, width, height.
        block_size = 20  # block size in pixels
        screen = pygame.Surface(disp_size)
        pygame.display.set_caption('Maze Generator / KS 2022')

        # intialize a maze, given size (y, x)
        self.maze = Maze(rect[2] // (block_size * 2) - 1, rect[3] // (block_size * 2) - 1)
        self.maze.screen = screen  # if this is set, the maze generation process will be displayed in a window. otherwise not.
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

    def makeCellImgMat(self):

        imgInfoMat = np.zeros(self.maze.blocks.shape+(4,))
        imgMat = np.zeros(self.maze.blocks.shape, dtype=object)
        notWallIdx = [0, 2, 3] # in self.blocks 0 is road 2 is character, and 3 is goal
        
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

    def draw(self):
        for row in range(self.imgMat.shape[0]):
            for col in range(self.imgMat.shape[1]):
                img = self.imgs[self.imgMat[row][col]]
                imgPos = (col * img.get_width(), row * img.get_height())
                self.screen.blit(img, imgPos)

        pygame.display.update()

if __name__=="__main__":
    env = MazeEnv_v0()
    running = True
    env.draw()

    while running:
        pass
