from Maze_Generator import *
from random import choice

class MazeEnv_v0():
    def __init__(self):
        # set screen size and initialize it
        pygame.display.init()
        disp_size = (500, 500)
        rect = np.array([0, 0, disp_size[0], disp_size[1]])  # the rect inside which to draw the maze. Top x, top y, width, height.
        block_size = 10  # block size in pixels
        screen = pygame.display.set_mode(disp_size)
        pygame.display.set_caption('Maze Generator / KS 2022')

        # intialize a maze, given size (y, x)
        self.maze = Maze(rect[2] // (block_size * 2) - 1, rect[3] // (block_size * 2) - 1)
        self.maze.screen = screen  # if this is set, the maze generation process will be displayed in a window. otherwise not.
        screen.fill((0, 0, 0))
        self.maze.screen_block_size = np.min(rect[2:4] / np.flip(self.maze.block_size))
        self.maze.screen_block_offset = rect[0:2] + (rect[2:4] - self.maze.screen_block_size * np.flip(self.maze.block_size)) // 2

        self.maze.gen_maze_2D()

        #=================================================================================
        # IMG PATH SETTING
        #=================================================================================

        import os.path as path
        root_dir = path.dirname(path.abspath(__file__)) + '/imgs/'

        self.ROAD_IMG = root_dir + "road.jpg"
        self.CHARACTER_IMG = root_dir + "character.jpg"
        self.GOAL_IMG = root_dir + "goal.jpg"

        self.wallName_to_Info = {
            # it was made to pick woods img by index
            # east, west, south, north wall information
            # 1: no tree, 2: only right(down) tree, 3: only left(up) tree, 4: two trees
            root_dir+"wall_01.jpg": [3, 3, 1, 4], 
            root_dir+"wall_02.jpg": [3, 3, 1, 4],
            root_dir+"wall_03.jpg": [1, 1, 4, 1],
            root_dir+"wall_04.jpg": [2, 2, 4, 1],
            root_dir+"wall_05.jpg": [1, 4, 3, 3],
            root_dir+"wall_06.jpg": [1, 4, 3, 3],
            root_dir+"wall_07.jpg": [1, 3, 1, 3],
            root_dir+"wall_08.jpg": [1, 2, 3, 1],
            root_dir+"wall_09.jpg": [1, 2, 3, 1],
            root_dir+"wall_10.jpg": [4, 4, 4, 4],
            root_dir+"wall_11.jpg": [1, 1, 1, 1],
            root_dir+"wall_12.jpg": [1, 1, 1, 1],
            root_dir+"wall_13.jpg": [4, 1, 2, 2],
            root_dir+"wall_14.jpg": [4, 1, 2, 2],
            root_dir+"wall_15.jpg": [3, 1, 1, 2],
            root_dir+"wall_16.jpg": [2, 1, 2, 1],
            root_dir+"wall_17.jpg": [2, 1, 2, 1],
        }

        #=================================================================================

        self.imgMat = self.makeCellImgMat()

    def makeCellImgMat(self):

        imgInfoMat = np.zeros(self.maze.blocks.shape+(4,))
        imgMat = np.zeros(self.maze.blocks.shape, dtype=object)
        
        for row in range(imgInfoMat.shape[0]):
            wallInfo = [0, 0, 0, 0]

            for col in range(imgInfoMat.shape[1]):
               
                if self.maze.blocks[row][col] == 1: # if its wall
                    wallInfo[0] = imgInfoMat[row][col+1][1] if col+1 < imgInfoMat.shape[1] else 0
                    wallInfo[1] = imgInfoMat[row][col-1][0] if col-1 > 0 else 0
                    wallInfo[2] = imgInfoMat[row+1][col][3] if row+1 < imgInfoMat.shape[0] else 0 
                    wallInfo[3] = imgInfoMat[row-1][col][2] if row-1 > 0 else 0

                    print(wallInfo)

                    wallName = self.wallInfoToName(wallInfo)
                    imgMat[row][col] = wallName
                    imgInfoMat[row][col] = self.wallName_to_Info[wallName]

                else: # if its road
                    imgMat[row][col] = self.ROAD_IMG

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
        for row in self.imgMat.shape[0]:
            for col in self.imgMat.shape[1]:
                imgPos = (row * self.maze.block_size, col * self.maze.block_size)
                img = pygame.image.load(self.imgMat[row][col])
                new_rect = rotated_image.get_rect(center = image.get_rect(topleft = top_left).center)
                rect = img.get_rect(topLeft = imgPos)
                win.blit(img, rect.topleft)
        
        self.screen.blit()
        

if __name__=="__main__":
    env = MazeEnv_v0()
    running = True

    while running:
        pass
