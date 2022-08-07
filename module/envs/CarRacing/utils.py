import pygame

def scale_image(img, factor):
    size = round(img.get_width() * factor), round(img.get_height() * factor)
    return pygame.transform.scale(img, size)

def blit_rotate_center(win, image, top_left, angle):

    rotated_image = pygame.transform.rotate(image, angle)
    new_rect = rotated_image.get_rect(center = image.get_rect(topleft = top_left).center)
    win.blit(rotated_image, new_rect.topleft)

    return new_rect

def blit_text_center(win, font, text):
    render = font.render(text, 1, (200, 200, 200))
    win.blit(render, (win.get_width()/2 - render.get_width()/2, win.get_height()/2 - render.get_height()/2))

# convert action type to int
def action_to_int(action):
    import torch

    if not type(action) in [torch.Tensor, list]:
        try:
            return action
        except:
            raise ValueError("action must be torch.Tensor or list")

    if type(action)==torch.Tensor:
        actionLst = action.tolist()
        if action.dim==1 and len(actionLst)==1:
            return actionLst[0]
        elif action.dim==0:
            return actionLst
        else:
            raise ValueError("Action Tensor dimension must be smaller than 2") 
            
    if type(action)==list:
        if len(action)==1:
            return action[0]
        else:
            raise ValueError("Action list dimension must be smaller than 2") 
   
    

