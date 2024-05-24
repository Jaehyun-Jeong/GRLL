from typing import Union, Tuple

import sys
sys.path.append("../../../")

from grll.envs.Car.Car import Car
from grll.utils.ActionSpace.ActionSpace import ActionSpace

import os.path as path
import pygame
import numpy as np
import torch


class CarEnv_base():

    WIDTH = 1980
    HEIGHT = 1080

    def __init__(
        self,
        difficulty: int,
        carSize: tuple = (60, 60),
    ):

        self.carSize = carSize

        if not difficulty in [1, 2, 3, 4, 5]:
            raise ValueError("Difficulty must be between 1 to 5!")

        self.screen = self.set_display_mode()

        # Set car in the map
        self.car = Car(carSize)

        # Load game_map image
        map_img_path = path.join(path.dirname(path.abspath(__file__)), f'map{difficulty}.png')
        self.game_map = pygame.image.load(map_img_path).convert() # Convert Speeds Up A Lot

        # isRender is needed for render option
        self.isRender = False

    def set_display_mode(self):

        try:
            screen = pygame.display.set_mode(
                    (self.WIDTH, self.HEIGHT), pygame.HIDDEN)
        except pygame.error:  # If there is no displaying device
            screen = pygame.Surface(
                    (self.WIDTH, self.HEIGHT))

        return screen

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
            if(self.car.speed >= 14):
                self.car.speed -= 2
        elif action == 3:
            # Speed Up
            self.car.speed += 2
        else:
            raise ValueError("Action out of bound!")

    def get_state(self) -> np.ndarray:
        return np.array(self.car.get_data(), dtype=np.float32)

    def get_done(self) -> bool:

        if not self.car.is_alive():
            return True
        else:
            return False


class CarEnv_v0(CarEnv_base):

    def __init__(
        self,
        difficulty: int = 1,
        carSize: tuple = (60, 60),
    ):

        super().__init__(
            carSize=carSize,
            difficulty=difficulty
        )

        # Car have 4 movements
        # 0: Left
        # 1: Right
        # 2: Slow Down
        # 3: Speed Up
        self.num_action = 4
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

        # Gymnasium info
        info = {}

        return next_state, reward, done, done, info

    def reset(self) -> np.ndarray:

        # Initialize pygame
        pygame.init()

        self.screen = self.set_display_mode()

        if pygame.display.get_active():
            pygame.display.set_mode((self.width, self.height), flags=pygame.hidden)

        self.car = Car(self.carSize)
        self.car.update(self.game_map)

        # it makes render option work
        self.isRender = False

        # Gymnasium info
        info = {}

        return self.get_state(), info

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
