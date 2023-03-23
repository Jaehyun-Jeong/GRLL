from copy import copy
import numpy as np

def policy_diagram(
        env,
        module,  # get action
        ):

    act_mark = {
            0: '→',  # east
            1: '←',  # west
            2: '↓',  # south
            3: '↑',  # north
            }
    
    # Set goal again
    # Because, if agent reach the goal, then env lose goal sign 
    env.blocks[1][1] = 3

    for row in range(env.blocks.shape[0]):
        line = ''
        for col in range(env.blocks.shape[1]):
            if env.blocks[row][col] == 1:  # wall
                line += '@'
            elif env.blocks[row][col] == 3:  # goal
                line += 'G'
            else:
                env.blocks[env.blocks == 2] = 0  # character to road
                env.blocks[row][col] = 2  # Set character in row, col
                action = module.value.get_action(
                        env.get_state(),
                        isTest=True)
                line += act_mark[action]
        print(line)
