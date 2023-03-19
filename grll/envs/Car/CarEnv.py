from typing import Union, Tuple

import sys
sys.path.append("../../../")

from grll.envs.Car.Car import Car
from grll.utils.ActionSpace.ActionSpace import ActionSpace

import pygame
import numpy as np
import torch


class CarEnv_base():

    WIDTH = 1980
    HEIGHT = 1080

    def __init__(self):

        try:
            self.screen = pygame.display.set_mode(
                    (self.WIDTH, self.HEIGHT), pygame.HIDDEN)
        except pygame.error:  # If there is no displaying device
            self.screen = pygame.Surface(
                    (self.WIDTH, self.HEIGHT))

        # Set car in the map
        self.car = Car()

        # Load game_map image
        self.game_map = pygame.image.load('map.png').convert() # Convert Speeds Up A Lot

        # count frame
        self.counter = 0

        # isRender is needed for render option
        self.isRender = False

    def move(
            self,
            action: int):

        if action == 0:
            # Left
            self.car.angle += 10
        elif action == 1:
            # Right
            self.car.angle -= 10
        elif action == 2:
            # Slow Down
            if(self.car.speed - 2 >= 12):
                self.car.speed -= 2
        elif action == 3:
            # Speed Up
            self.car.speed += 2
        else:
            raise ValueError("Action out of bound!")

        self.counter += 1

    def get_state(self) -> np.ndarray:
        return np.array(self.car.get_data(), dtype=np.float32)

    def get_done(self) -> bool:

        if (not self.car.is_alive() or self.counter == 30 * 40):
            return True
        else:
            return False


class CarEnv_v0(CarEnv_base):

    def __init__(self):

        super().__init__()

        # Car have 4 movements
        # 0: Left
        # 1: Right
        # 2: Slow Down
        # 3: Speed Up
        self.num_actions = 4
        self.num_obs = 5  # Car has 5 lazers

        self.action_space = ActionSpace(
                high=np.array([3]),
                low=np.array([0]))

    # Returns next_state, reward, done, action
    def step(self, action: Union[int, torch.Tensor]) \
            -> Tuple[np.ndarray, float, bool, torch.Tensor]:

        self.move(action)
        next_state = self.get_state()
        done = self.get_done()
        reward = self.car.get_reward()

        self.car.update(self.game_map)
        
        return next_state, reward, done, action

    def reset(self) -> np.ndarray:

        self.car = Car()
        self.car.update(self.game_map)

        if pygame.display.get_active():
            pygame.display.set_mode((self.WIDTH, self.HEIGHT), flags=pygame.HIDDEN)


        return self.get_state()

    def render(self):
        try:
            if not self.isRender:
                pygame.display.set_mode((self.WIDTH, self.HEIGHT), flags=pygame.SHOWN)
                self.isRender = True
            self.screen.blit(self.game_map, (0, 0))
            self.car.draw(self.screen)
            pygame.display.flip()
        except:
            raise RuntimeError("No available display to render")

    def close(self):
        pygame.quit()


if __name__ == "__main__":

    import random

    env = CarEnv_v0()
    state = env.reset()
    while True:
        #env.render()
        #action = random.randint(0, 3)
        action = 2
        state, reward, done, _ = env.step(action)

        print(state)
        print(reward)
        print(done)
        print("====================================")
        
        if done:
            break

    env.close()
