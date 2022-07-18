from main import *

class linetrace:

    def __init__(self):
        self.LINE_COUNT = 8
        self.LINE_LEN = 100
        self.GREEN = (0, 255, 0)
        
    def draw(self, win, start_pos, degree):

        for lineNum in range(self.LINE_COUNT):

            radian = math.radians(degree + lineNum*45)
            end_x = start_pos[0] + self.LINE_LEN * -math.sin(radian)
            end_y = start_pos[1] + self.LINE_LEN * -math.cos(radian)
            end_pos = (end_x, end_y)

            pygame.draw.line(win, self.GREEN, start_pos, end_pos, 3)

def env_collision(player_car, computer_car, game_info):
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
            pygame.time.wait(5000)
            game_info.reset()
            player_car.reset()
            computer_car.reset()
        else:
            game_info.next_level()
            player_car.reset()
            computer_car.next_level(game_info.level)

def get_center_pos(player_car):
    rect = player_car.rect
    x = (rect.topleft[0] - rect.bottomright[0])/2 + rect.bottomright[0]
    y = (rect.topleft[1] - rect.bottomright[1])/2 + rect.bottomright[1]

    return x, y

def draw_env(win, images, player_car, computer_car, game_info, linetrace):
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

    linetrace.draw(WIN, get_center_pos(player_car), player_car.angle)
    pygame.display.update()

run = True
images = [(GRASS, (0, 0)), (TRACK, (0, 0)), (FINISH, FINISH_POSITION), (TRACK_BORDER, (0,0))]
player_car = PlayerCar(4, 4)
computer_car = ComputerCar(2, 4, PATH)
game_info = GameInfo()
line = linetrace()

while run:
    
    draw_env(WIN, images, player_car, computer_car, game_info, line)

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

    env_collision(player_car, computer_car, game_info)

    if game_info.game_finished():
        blit_text_center(win, MAIN_FONT, "You won the game!")
        pygame.time.wait(5000)
        game_info.reset()
        player_car.reset()
        computer_car.reset()

pygame.quit()
