from collections import deque
import torch
import numpy as np
try:
    from main import *
except:
    from module.envs.CarRacing.main import *

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
                test_pos = (line_mag*end_pos[0]+start_pos[0], line_mag*end_pos[1]+start_pos[1])
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

def draw_env(win, images, player_car, game_info, computer_car=None, lines=None):
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
    
    def __init__(self, ExploringStarts=False):

        self.images = [(GRASS, (0, 0)), (TRACK, (0, 0)), (FINISH, FINISH_POSITION), (TRACK_BORDER, (0,0))]
        self.player_car = PlayerCar(4, 4)
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

    def step(self, action: torch.tensor):

        self.move(action)

        next_state = self.get_state()
        done, reward = self.__done_reward()

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
            start_pos, start_angle = self.__get_random_pos_angle()
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
    def __get_random_pos_angle():
        
        import random
            
        PATH = [(150, 200), (175, 119), (110, 70), (56, 133), (70, 481), (318, 710), (404, 680), (418, 521), (507, 475), (600, 551), (613, 715), (736, 690),
                (734, 399), (611, 357), (409, 343), (433, 257), (697, 230), (738, 123), (581, 71), (303, 78), (275, 377), (176, 388)]
        ANGLE = [0, 45, 90, 180, -135, -90, 0, -45, -90, -180, -134, 0, 45, 90, 45, -90, -45, 45, 90, 135, 135, 0]

        rand_ind = random.randint(0, len(PATH)-1)

        return PATH[rand_ind], ANGLE[rand_ind]

    def move(self, action: torch.tensor):

        if action == 0: # left
            self.player_car.rotate(left=True)
        elif action == 1: # center
            pass
        elif action == 2: # right 
            self.player_car.rotate(right=True)
        else:
            raise ValueError("Action is out of bound!")

        self.player_car.move_forward()

    def get_state(self): # use collided distances as state

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
    def __line_preprocess(self, dist):
        return dist / self.lines.LINE_LEN

    def __done_reward(self):

        done = False
        reward = -0.04 # Every step it have -0.04 reward 
        
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

    def __init__(self, stackSize: int=4, ExploringStarts: bool=False):

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
        self.__init_stackedStates(self.stackSize)

        # Number of actions and observations
        # There are 3 actions left, center, and right, then 8 lines to check distance to wall
        self.num_actions = 3
        self.num_obs = self.reset().shape

        #=========================================================

    def step(self, action: torch.tensor):
        # 3 frames per move
        self.move(action)
        draw_env(WIN, self.images, self.player_car, self.game_info, lines=self.lines)
        self.move(action)
        draw_env(WIN, self.images, self.player_car, self.game_info, lines=self.lines)
        return super().step(action)

    def reset(self):
        
        self.isRender = False
        self.game_info.reset()

        if self.ExploringStarts:
            start_pos, start_angle = self.__get_random_pos_angle()
            self.player_car.START_POS = start_pos
            self.player_car.reset(start_angle)
        else:
            self.player_car.reset()

        if pygame.display.get_active():
            pygame.display.set_mode((WIDTH, HEIGHT), flags=pygame.HIDDEN)

        draw_env(WIN, self.images, self.player_car, self.game_info, lines=self.lines)

        self.__init_stackedStates(self.stackSize)
        initial_state = self.get_state()
        
        return initial_state
        
    def get_state(self):

        screen = pygame.surfarray.pixels3d(WIN) # game screen img to numpy ndarray(RGB)
        screen = self.__grayscale(screen)
        self.stackedStates.append(screen) # from RGB to grayscale img
        
        state = torch.from_numpy(np.array(self.stackedStates)).unsqueeze(0)
        state = self._transforms(state).squeeze(0)

        return state.to(torch.float).numpy()

    def __init_stackedStates(self, frames: int):
        draw_env(WIN, self.images, self.player_car, self.game_info, lines=self.lines)

        screen = pygame.surfarray.pixels3d(WIN) # game screen img to numpy ndarray(RGB)
        screen = self.__grayscale(screen) # from RGB to grayscale img
        
        self.stackedStates.extend([screen]*frames)

    @staticmethod
    def __grayscale(numpy_array: np.ndarray):
        return np.dot(numpy_array[..., :3], [0.299, 0.587, 0.114])

class RacingEnv_v3(RacingEnv_v0):

    def __init__(self, stackSize: int=4, ExploringStarts: bool=False):

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

        self._stateImgSize = (150, 150)

        self._transforms = T.Compose(
            [T.Resize(self._stateImgSize), T.Normalize(0, 255)]
        )

        self.stackedStates = deque([], maxlen=self.stackSize)
        self.__init_stackedStates(self.stackSize)

        # Number of actions and observations
        # There are 3 actions left, center, and right, then 8 lines to check distance to wall
        self.num_actions = 3
        self.num_obs = self.reset().shape

        #=========================================================
        
    def step(self, action: torch.tensor):
        # 3 frames per move
        self.move(action)
        draw_env(WIN, self.images, self.player_car, self.game_info, lines=self.lines)
        self.move(action)
        draw_env(WIN, self.images, self.player_car, self.game_info, lines=self.lines)
        return super().step(action)

    def reset(self):
        
        self.isRender = False
        self.game_info.reset()

        if self.ExploringStarts:
            start_pos, start_angle = self.__get_random_pos_angle()
            self.player_car.START_POS = start_pos
            self.player_car.reset(start_angle)
        else:
            self.player_car.reset()

        if pygame.display.get_active():
            pygame.display.set_mode((WIDTH, HEIGHT), flags=pygame.HIDDEN)

        draw_env(WIN, self.images, self.player_car, self.game_info, lines=self.lines)

        self.__init_stackedStates(self.stackSize)
        initial_state = self.get_state()
        
        return initial_state

    def get_state(self):

        # Getting self.player_car.rect at least one draw_env execution
        center = self.player_car.rect.center
        leftTop = (center[0]-self._stateImgSize[0]/2, center[1]-self._stateImgSize[1]/2)

        screen = WIN.subsurface(leftTop[0], leftTop[1], self._stateImgSize[0], self._stateImgSize[1])
        screen = pygame.surfarray.pixels3d(screen) # game screen img to numpy ndarray(RGB)
        screen = self.__grayscale(screen) # from RGB to grayscale img

        self.stackedStates.append(screen)

        state = torch.from_numpy(np.array(self.stackedStates)).unsqueeze(0)
        state = self._transforms(state).squeeze(0)

        return state.to(torch.float).numpy()

    def __init_stackedStates(self, frames: int):
        
        # Getting self.player_car.rect at least one draw_env execution
        draw_env(WIN, self.images, self.player_car, self.game_info, lines=self.lines)
        center = self.player_car.rect.center
        leftTop = (center[0]-self._stateImgSize[0]/2, center[1]-self._stateImgSize[1]/2)

        screen = WIN.subsurface(leftTop[0], leftTop[1], self._stateImgSize[0], self._stateImgSize[1])
        screen = pygame.surfarray.pixels3d(screen) # game screen img to numpy ndarray(RGB)
        screen = self.__grayscale(screen) # from RGB to grayscale img
        
        self.stackedStates.extend([screen]*frames)

    @staticmethod
    def __grayscale(numpy_array: np.ndarray):
        return np.dot(numpy_array[..., :3], [0.299, 0.587, 0.114])

if __name__=="__main__":
    from main import *
    import matplotlib.pyplot as plt

    RacingEnv = RacingEnv_v2()

    for episode in range(3):

        state = RacingEnv.reset()
        RacingEnv.render()

        for i in range(1000):

            fig = plt.figure()
            ax1 = fig.add_subplot(4, 1, 1)
            ax1.imshow(state[0], cmap="gray")
            ax2 = fig.add_subplot(4, 1, 2)
            ax2.imshow(state[1], cmap="gray")
            ax3 = fig.add_subplot(4, 1, 3)
            ax3.imshow(state[2], cmap="gray")
            ax4 = fig.add_subplot(4, 1, 4)
            ax4.imshow(state[3], cmap="gray")
            plt.show()

            state, reward, done, _ = RacingEnv.step(episode)
            if done:
                break

    RacingEnv.close()
    '''
    for i in range(1000):
        print(RacingEnv.step(1))

    RacingEnv.close()
    '''
