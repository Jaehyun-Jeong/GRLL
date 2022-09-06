from collections import deque
import torch
import numpy as np
import math
import pygame

if __name__=="__main__":
    from main import *
    from utils import action_to_int, check_actionList
else:
    from module.envs.CarRacing.main import *
    from module.envs.CarRacing.utils import action_to_int, check_actionList


class Lines():

    def __init__(self, win):
        self.LINE_LEN = 100
        self.GREEN = (0, 255, 0)
        self.BLUE = (0, 0, 255)

        # 8 lines
        self.DEGREES = [45*i for i in range(8)]
        self.hit_points = deque([], maxlen=8)

        self.surface = win

    def draw(self, win, mask, start_pos, degree):

        for line_degree in self.DEGREES:

            radian = math.radians(degree + line_degree)

            end_x = self.LINE_LEN * -math.sin(radian)
            end_y = self.LINE_LEN * -math.cos(radian)
            end_pos = (end_x, end_y)

            is_hit = False
            for length_mag in range(1000):
                line_mag = (length_mag+1) / 1000
                test_pos = (line_mag*end_pos[0]+start_pos[0],
                            line_mag*end_pos[1]+start_pos[1])

                if mask.get_at(test_pos):
                    end_pos = test_pos
                    is_hit = True
                    break

            if not is_hit:
                end_pos = (end_x+start_pos[0], end_y+start_pos[1])
            else:
                pygame.draw.circle(self.surface, self.BLUE, end_pos, 3)

            self.hit_points.append(end_pos)
            pygame.draw.line(self.surface, self.GREEN, start_pos, end_pos, 1)

        win.blit(self.surface, self.surface.get_rect())

    def collide(self):
        return list(self.hit_points)


def draw_env(win: pygame.Surface,
             images: list[pygame.Surface],
             player_car: PlayerCar,
             game_info: GameInfo,
             computer_car: ComputerCar= None,
             lines: Lines = None):

    for img, pos in images:
        win.blit(img, pos)

    player_car.draw(win)

    if computer_car != None:
        computer_car.draw(win)

    if lines != None: 
        lines.draw(win, TRACK_BORDER_MASK, player_car.rect.center, player_car.angle)

    if pygame.display.get_active():
        pygame.display.update()

class RacingEnv_v0():
    
    def __init__(self, ExploringStarts: bool = False):

        self.images = [(GRASS, (0, 0)), (TRACK, (0, 0)), (FINISH, FINISH_POSITION), (TRACK_BORDER, (0,0))]

        # player car to full velocity
        self.player_car = PlayerCar(4, 4)
        self.player_car.vel = self.player_car.max_vel

        self.game_info = GameInfo()
        self.start_pos = (150, 200)
        self.start_angle = 0
        self.ExploringStarts = ExploringStarts
        self.isRender = False

        # make line
        self.lines = Lines(WIN)

        # Number of actions and observations
        # There are 3 actions left, center, and right, then 8 lines to check distance to wall
        self.num_actions = 3
        self.num_obs = 8

        self.game_info.start_level()
        
        # reward
        self._reward = 0

    def step(self, action: torch.Tensor) \
            -> tuple[np.ndarray, float, bool, torch.Tensor]:

        action = action_to_int(action)
        self.move(action)

        next_state = self.get_state()
        done, reward = self.done_reward()

        draw_env(WIN, self.images, self.player_car, self.game_info, lines=self.lines)

        return next_state, reward, done, action

    def render(self):
        try:
            pygame.display.set_mode((WIDTH, HEIGHT), flags=pygame.SHOWN)
            if not self.isRender:
                self.isRender = True
                draw_env(WIN, self.images, self.player_car, self.game_info, lines=self.lines)
        except:
            raise RuntimeError("No available display to render")

    def reset(self):
        
        self.game_info.reset()

        if self.ExploringStarts:
            start_pos, start_angle = self.get_random_pos_angle()
            self.player_car.START_POS = start_pos
            self.player_car.reset(start_angle)
        else:
            self.player_car.reset()

        if pygame.display.get_active():
            pygame.display.set_mode((WIDTH, HEIGHT), flags=pygame.HIDDEN)

        draw_env(WIN, self.images, self.player_car, self.game_info, lines=self.lines)

        initial_state = self.get_state()
    
        return initial_state

    def close(self):
        pygame.quit()
    
    @staticmethod
    def get_random_pos_angle() -> tuple[tuple[int, int], int] :
        
        import random
            
        PATH = [(150, 200), (175, 119), (110, 70), (56, 133), (70, 481), (318, 710), (404, 680), (418, 521), (507, 475), (600, 551), (613, 715), (736, 690),
                (734, 399), (611, 357), (409, 343), (433, 257), (697, 230), (738, 123), (581, 71), (303, 78), (275, 377), (176, 388)]
        ANGLE = [0, 45, 90, 180, -135, -90, 0, -45, -90, -180, -134, 0, 45, 90, 45, -90, -45, 45, 90, 135, 135, 0]

        rand_ind = random.randint(0, len(PATH)-1)

        return PATH[rand_ind], ANGLE[rand_ind]

    def move(self, action: torch.Tensor):

        if action == 0: # left
            self.player_car.rotate(left=True)
        elif action == 1: # center
            pass
        elif action == 2: # right 
            self.player_car.rotate(right=True)
        else:
            raise ValueError("Action is out of bound!")

        self.player_car.move_forward()

    def get_state(self) -> list[float]: # use collided distances as state

        # line collider
        hit_points = self.lines.collide()
        dists = []
        for hit_point in hit_points:
            car_center = self.player_car.rect.center
            relative_point = (hit_point[0]-car_center[0], hit_point[1]-car_center[1])
            dist = (relative_point[0]**2 + relative_point[1]**2)**(1/2)
            dists.append(self.__line_preprocess(dist))

        return dists

    # dists length from 0 to 1
    def __line_preprocess(self, dist: float) -> float:
        return dist / self.lines.LINE_LEN

    def done_reward(self) -> tuple[bool, float]:

        done = False
        reward = 0.004 # Every step it have -0.04 reward 
        
        if self.player_car.collide(TRACK_BORDER_MASK) != None:
            done = True
            reward -= 1

        player_finish_poi_collide = self.player_car.collide(FINISH_MASK, *FINISH_POSITION)
        if player_finish_poi_collide != None:
            done = True
            if player_finish_poi_collide[1] == 0:
                reward -= 1
            else:
                reward += 1
        
        return done, reward

class RacingEnv_v2(RacingEnv_v0):

    def __init__(self,
                 stackSize: int=4,
                 ExploringStarts: bool=False):

        super().__init__(
            ExploringStarts=ExploringStarts
        )

        self.lines = None # delete the line
        self.stackSize = stackSize

        #=========================================================
        # STACKED GRAYSCALE IMG DATA
        #=========================================================

        from collections import deque
        from torchvision import transforms as T

        self._transforms = T.Compose(
            [T.Resize((84, 84)), T.Normalize(0, 255)]
        )

        self.stackedStates = deque([], maxlen=self.stackSize)
        self.init_stackedStates(self.stackSize)

        # Number of actions and observations
        # There are 3 actions left, center, and right, then 8 lines to check distance to wall
        self.num_actions = 3
        self.num_obs = self.reset().shape

        #=========================================================

    def step(self, action: torch.Tensor) \
            -> tuple[np.ndarray, float, bool, torch.Tensor]:

        # 3 frames per move
        action = action_to_int(action)
        self.move(action)
        draw_env(WIN, self.images, self.player_car, self.game_info, lines=self.lines)
        self.move(action)
        draw_env(WIN, self.images, self.player_car, self.game_info, lines=self.lines)

        return super().step(action)

    def reset(self) -> np.ndarray:
        
        self.isRender = False
        self.game_info.reset()

        if self.ExploringStarts:
            start_pos, start_angle = self.get_random_pos_angle()
            self.player_car.START_POS = start_pos
            self.player_car.reset(start_angle)
        else:
            self.player_car.reset()

        if pygame.display.get_active():
            pygame.display.set_mode((WIDTH, HEIGHT), flags=pygame.HIDDEN)

        draw_env(WIN, self.images, self.player_car, self.game_info, lines=self.lines)

        self.init_stackedStates(self.stackSize)
        initial_state = self.get_state()
        
        return initial_state
        
    def get_state(self) -> np.ndarray:

        screen = pygame.surfarray.pixels3d(WIN) # game screen img to numpy ndarray(RGB)
        screen = self.grayscale(screen)
        self.stackedStates.append(screen) # from RGB to grayscale img

        state = torch.from_numpy(np.array(self.stackedStates))
        state = self._transforms(state)

        return state.to(torch.float).numpy()

    def init_stackedStates(self, frames: int):
        draw_env(WIN, self.images, self.player_car, self.game_info, lines=self.lines)

        screen = pygame.surfarray.pixels3d(WIN) # game screen img to numpy ndarray(RGB)
        screen = self.grayscale(screen) # from RGB to grayscale img
        
        self.stackedStates.extend([screen]*frames)

    @staticmethod
    def grayscale(numpy_array: np.ndarray) -> np.ndarray:
        return np.dot(numpy_array[..., :3], [0.299, 0.587, 0.114])

class RacingEnv_v3(RacingEnv_v0):

    def __init__(
            self,
            stackSize: int=4,
            ExploringStarts: bool=False,
            skipFrame: int=2,
            imgSize: tuple=(86, 86), 
            isFlatten: bool=False):

        super().__init__(
            ExploringStarts=ExploringStarts
        )

        self.lines = None # delete the line
        self.stackSize = stackSize
        self.skipFrame = skipFrame
        self.isFlatten = isFlatten

        #=========================================================
        # STACKED GRAYSCALE IMG DATA
        #=========================================================

        from collections import deque
        from torchvision import transforms as T

        self._sliceImgSize = (150, 150)
        self._stateImgSize = imgSize

        self._transforms = T.Compose(
            [T.Resize(self._stateImgSize), T.Normalize(0, 255)]
        )

        self.stackedStates = deque([], maxlen=self.stackSize)
        self.init_stackedStates(self.stackSize)

        # Number of actions and observations
        # There are 3 actions left, center, and right, then 8 lines to check distance to wall
        self.num_actions = 3
        self.num_obs = self.reset().shape

        #=========================================================
        
    def step(self, action: torch.Tensor) \
            -> tuple[np.ndarray, float, bool, torch.Tensor]:

        # self.skipFrame per move
        action = action_to_int(action)
        for _ in range(self.skipFrame - 1):  # -1 for last step
            self.move(action)
            draw_env(WIN, self.images, self.player_car, self.game_info, lines=self.lines)

        return super().step(action)

    def reset(self) -> np.ndarray:
        
        self.isRender = False
        self.game_info.reset()

        if self.ExploringStarts:
            start_pos, start_angle = self.get_random_pos_angle()
            self.player_car.START_POS = start_pos
            self.player_car.reset(start_angle)
        else:
            self.player_car.reset()

        if pygame.display.get_active():
            pygame.display.set_mode((WIDTH, HEIGHT), flags=pygame.HIDDEN)

        draw_env(WIN, self.images, self.player_car, self.game_info, lines=self.lines)

        self.init_stackedStates(self.stackSize)
        initial_state = self.get_state()
        
        return initial_state

    def crop_img_numpy(self) -> np.ndarray:

        # Getting self.player_car.rect at least one draw_env execution
        center = self.player_car.rect.center
        leftTop = (center[0]-self._sliceImgSize[0]/2, center[1]-self._sliceImgSize[1]/2)

        sizeLst = [self._sliceImgSize[0], self._sliceImgSize[1]]
        if leftTop[0] < 0:
            sizeLst.append(center[0]*2)
        if leftTop[0]+self._sliceImgSize[0] > WIN.get_width():
            sizeLst.append(2*(WIN.get_width()-center[0]))
        if leftTop[1] < 0:
            sizeLst.append(center[1]*2)
        if leftTop[1]+self._sliceImgSize[1] > WIN.get_height():
            sizeLst.append(2*(WIN.get_height()-center[1]))

        sliceSize = min(sizeLst)
        leftTop = (center[0]-sliceSize/2, center[1]-sliceSize/2)

        screen = WIN.subsurface(leftTop[0], leftTop[1], sliceSize, sliceSize)
        screen = pygame.surfarray.pixels3d(screen) # game screen img to numpy ndarray(RGB)
        screen = self.grayscale(screen) # from RGB to grayscale img

        if sliceSize < min(self._sliceImgSize):
            from PIL import Image
            screen = np.array(Image.fromarray(screen).resize(self._sliceImgSize))

        return screen

    def get_state(self) -> np.ndarray:
        self.stackedStates.append(self.crop_img_numpy())

        state = torch.from_numpy(np.array(self.stackedStates))
        state = self._transforms(state)

        if self.isFlatten:
            state = torch.flatten(state)

        return state.to(torch.float).numpy()

    def init_stackedStates(self, frames: int):

        draw_env(WIN, self.images, self.player_car, self.game_info, lines=self.lines)
        self.stackedStates.extend([self.crop_img_numpy()]*frames)

    @staticmethod
    def grayscale(numpy_array: np.ndarray) -> np.ndarray:
        return np.dot(numpy_array[..., :3], [0.299, 0.587, 0.114])

class RacingEnv_v4(RacingEnv_v3):
        
    def step(self, action: torch.Tensor) \
            -> tuple[np.ndarray, float, bool, torch.Tensor]:

        if check_actionList(action):
            self.move(action)
            draw_env(WIN, self.images, self.player_car, self.game_info, lines=self.lines)
            self.move(action)

            next_state = self.get_state()
            done, reward = self.done_reward()

            draw_env(WIN, self.images, self.player_car, self.game_info, lines=self.lines)

            return next_state, reward, done, action 

    def move(self, action: torch.Tensor):
        if check_actionList(action):

            if action[0] == 0: # no accel
                self.player_car.reduce_speed()
            elif action[0] == 1: # accel
                self.player_car.move_forward()
            else:
                raise ValueError("Action is out of bound!")

            if action[1] == 0: # turn left
                self.player_car.rotate(left=True)
            elif action[1] == 1: # center
                pass
            elif action[1] == 2: # turn right
                self.player_car.rotate(right=True)
            else:
                raise ValueError("Action is out of bound!")

if __name__=="__main__":
    from random import choice

    RacingEnv = RacingEnv_v3(
            ExploringStarts=True,
            isFlatten=True)

    for episode in range(3):

        state = RacingEnv.reset()
        RacingEnv.render()

        for i in range(1000):

            print(state.shape)

            action = choice([0, 1, 2])
            action = torch.tensor(action)
            state, reward, done, _ = RacingEnv.step(action)

            if done:
                break

    RacingEnv.close()
