from collections import deque
import torch
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

def draw_env(win, images, player_car, computer_car, game_info, lines):
    for img, pos in images:
        win.blit(img, pos)

    level_text = MAIN_FONT.render(f"Level {game_info.level}", 1, (255, 255, 255))
    win.blit(level_text, (10, HEIGHT - level_text.get_height() - 70))

    time_text = MAIN_FONT.render(f"Time; {int(game_info.get_level_time())}s", 1, (255, 255, 255))
    win.blit(time_text, (10, HEIGHT - level_text.get_height() - 40))

    vel_text = MAIN_FONT.render(f"Vel; {round(player_car.vel, 1)}px/s", 1, (255, 255, 255))
    win.blit(vel_text, (10, HEIGHT - level_text.get_height() - 10))

    player_car.draw(win)
    computer_car.draw(win)
    
    lines.draw(win, TRACK_BORDER_MASK, player_car.rect.center, player_car.angle)

    if pygame.display.get_active():
        pygame.display.update()

class RacingEnv_v0():
    
    def __init__(self):

        self.run = True
        self.images = [(GRASS, (0, 0)), (TRACK, (0, 0)), (FINISH, FINISH_POSITION), (TRACK_BORDER, (0,0))]
        self.player_car = PlayerCar(4, 4)
        self.computer_car = ComputerCar(2, 4, PATH)
        self.game_info = GameInfo()
        self.start_pos = (150, 200)
        self.start_angle = 0

        # make line
        self.lines = Lines(WIN)

        # Number of actions and observations
        # There are 2 actions left and right, and 8 lines to check distance to wall
        self.num_actions = 2
        self.num_obs = 8

        self.game_info.start_level()
        draw_env(WIN, self.images, self.player_car, self.computer_car, self.game_info, self.lines)
        
        # reward
        self._reward = 0

    def step(self, action: torch.tensor):

        self.__move(action)
        self.computer_car.move()

        next_state = self.__line_collide()
        done, reward = self.__done_reward()

        draw_env(WIN, self.images, self.player_car, self.computer_car, self.game_info, self.lines)

        return next_state, reward, done, action 

    def render(self):
        try:
            pygame.display.set_mode((WIDTH, HEIGHT), flags=pygame.SHOWN)
        except:
            raise RuntimeError("No available display to render")

    def reset(self, exploring_starts = False):
        
        self.game_info.reset()
        self.computer_car.reset()

        if exploring_starts:
            start_pos, start_angle = self.__get_random_pos_angle()
            self.player_car.START_POS = start_pos
            self.player_car.reset(start_angle)
        else:
            self.player_car.reset()

        if pygame.display.get_active():
            pygame.display.set_mode((WIDTH, HEIGHT), flags=pygame.HIDDEN)
            pygame.display.update()

        initial_state = self.__line_collide()
        
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

    def __move(self, action: torch.tensor):

        if action == 0: # left
            self.player_car.rotate(left=True)
        elif action == 1: # right 
            self.player_car.rotate(right=True)
        else:
            raise ValueError("Action is out of bound!")

        self.player_car.move_forward()

    def __line_collide(self):

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
        reward = 0.004 # Every step it have -0.04 reward 
        
        if self.player_car.collide(TRACK_BORDER_MASK) != None:
            done = True
            reward -= 1

        computer_finish_poi_collide = self.computer_car.collide(FINISH_MASK, *FINISH_POSITION)
        if computer_finish_poi_collide != None:
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

if __name__=="__main__":
    RacingEnv = RacingEnv_v0()

    for i in range(1000):
        print(RacingEnv.step(0))

    RacingEnv.reset()

    for i in range(1000):
        print(RacingEnv.step(1))

    RacingEnv.close()
