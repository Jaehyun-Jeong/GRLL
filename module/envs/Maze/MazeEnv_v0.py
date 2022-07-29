from Maze_Generator import *

# set screen size and initialize it
pygame.display.init()
disp_size = (1920, 1080)
rect = np.array([0, 0, disp_size[0], disp_size[1]])  # the rect inside which to draw the maze. Top x, top y, width, height.
block_size = 10  # block size in pixels
screen = pygame.display.set_mode(disp_size)
pygame.display.set_caption('Maze Generator / KS 2022')
running = True

while running:

    # intialize a maze, given size (y, x)
    maze = maze(rect[2] // (block_size * 2) - 1, rect[3] // (block_size * 2) - 1)
    maze.screen = screen  # if this is set, the maze generation process will be displayed in a window. otherwise not.
    screen.fill((0, 0, 0))
    maze.screen_size = np.asarray(disp_size)
    maze.screen_block_size = np.min(rect[2:4] / np.flip(maze.block_size))
    maze.screen_block_offset = rect[0:2] + (rect[2:4] - maze.screen_block_size * np.flip(maze.block_size)) // 2
