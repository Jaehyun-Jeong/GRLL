from main import *
from collections import deque

class lines():

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

def env_collision(player_car, computer_car, game_info, lines):

    # line collider
    hit_points = lines.collide()
    dists = []
    for hit_point in hit_points:
        car_center = player_car.rect.center
        relative_point = (hit_point[0]-car_center[0], hit_point[1]-car_center[1])
        dists.append(round((relative_point[0]**2 + relative_point[1]**2)**(1/2),2))

    print(dists)
    print("==================================================================")

    if player_car.collide(TRACK_BORDER_MASK) != None:
        blit_text_center(WIN, MAIN_FONT, "You lost!")
        pygame.display.update()
        game_info.reset()
        player_car.reset()
        computer_car.reset()

    computer_finish_poi_collide = computer_car.collide(FINISH_MASK, *FINISH_POSITION)
    if computer_finish_poi_collide != None:
        blit_text_center(WIN, MAIN_FONT, "You lost!")
        pygame.display.update()
        game_info.reset()
        player_car.reset()
        computer_car.reset()
        
    player_finish_poi_collide = player_car.collide(FINISH_MASK, *FINISH_POSITION)
    if player_finish_poi_collide != None:
        if player_finish_poi_collide[1] == 0:
            blit_text_center(WIN, MAIN_FONT, "You lost!")
            pygame.display.update()
            game_info.reset()
            player_car.reset()
            computer_car.reset()
        else:
            game_info.next_level()
            player_car.reset()
            computer_car.next_level(game_info.level)

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

    '''
    #===============================================================================
    #TEST

    # line collider
    hit_point = lines.collide(TRACK_BORDER_MASK)
    if hit_point != None:
        relative_point = (hit_point[0]-player_car.x, hit_point[1]-player_car.y)
        dist = (relative_point[0]**2 + relative_point[1]**2)**(1/2)

        print(hit_point)
        print(dist)
        print(type(dist))
        print("==================================================================")

        pygame.draw.circle(win, (255, 0, 0), hit_point, 5)

    #===============================================================================
    '''

    pygame.display.update()

run = True
images = [(GRASS, (0, 0)), (TRACK, (0, 0)), (FINISH, FINISH_POSITION), (TRACK_BORDER, (0,0))]
player_car = PlayerCar(4, 4)
computer_car = ComputerCar(2, 4, PATH)
game_info = GameInfo()

# make line
lines = lines(WIN)


while run:
    
    draw_env(WIN, images, player_car, computer_car, game_info, lines)

    while not game_info.started:
        blit_text_center(WIN, MAIN_FONT, f"Press any key to start level {game_info.level}!")
        pygame.display.update()
        game_info.start_level()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False
            break

    move_player(player_car)
    computer_car.move()

    env_collision(player_car, computer_car, game_info, lines)

    if game_info.game_finished():
        blit_text_center(win, MAIN_FONT, "You won the game!")
        pygame.time.wait(5000)
        game_info.reset()
        player_car.reset()
        computer_car.reset()

pygame.quit()
