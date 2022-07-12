import pygame
from utils.game import Player
from utils.models import GestureRecognizer
import pickle
import cv2


# Game parameters
window_size = [500, 500]
delay = 1
player_width, player_height = 50, 50
player_acc = 0.7  # divided by fps -> doesn't actually work
player_color = [203, 96, 21]
gesture_control = True

# Initialize recognizer and capture
class_map = {
    0: 'Left',
    1: 'Right'
}
with open('saved_models/clf.pkl', 'rb') as f:
    clf = pickle.load(f)
recognizer = GestureRecognizer(saved_clf=clf)
capture = cv2.VideoCapture(0)

# Initialize pygame
pygame.init()
window = pygame.display.set_mode(window_size)
pygame.display.set_caption('Prototype')
clock = pygame.time.Clock()


run, ret = True, True
draw_fps = False

fps_font = pygame.font.SysFont(None, 16)
action_font = pygame.font.SysFont(None, 32)

player = Player(
    x=window_size[0] // 2 - player_width // 2,
    y=window_size[1] // 2 - player_height // 2,
    width=player_width, height=player_height,
    ax=player_acc, window=window_size
)

while run and ret:
    pygame.time.delay(delay)

    if gesture_control:
        # Retrieve and process webcam image
        ret, frame = capture.read()
        frame = cv2.resize(frame, (224, 224))
        frame = frame[:, ::-1, ::-1]

        # Model prediction and player action
        pred = recognizer.predict_image(frame)
        if pred is not None:
            left_pressed = pred == 0
            right_pressed = pred == 1
            action_text = class_map[pred]
        else:
            left_pressed = right_pressed = False
            action_text = 'Neutral'
    else:
        # Handle key presses
        keys = pygame.key.get_pressed()
        left_pressed, right_pressed = False, False

        if keys[pygame.K_LEFT]:
            left_pressed = True
            action_text = 'Left'
        if keys[pygame.K_RIGHT]:
            right_pressed = True
            action_text = 'Right'

        if left_pressed and right_pressed:
            action_text = 'Neutral'

    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False
            break
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                draw_fps = not draw_fps

    # Update player position
    x, y = player.update_pos(clock, left_pressed, right_pressed)

    # Render game
    window.fill((255, 255, 255))
    pygame.draw.rect(window, player_color, (x, y, player.width, player_height))

    action_image = action_font.render(action_text, True, player_color, (255, 255, 255))
    _, action_height = action_font.size(action_text)
    if action_image is not None:
        window.blit(action_image, (20, window_size[1] - action_height - 20))

    if draw_fps:
        text = 'FPS: ' + str(int(clock.get_fps()))
        fps_image = fps_font.render(text, True, (255, 255, 255), (0, 0, 0))
        window.blit(fps_image, (20, 20))

    pygame.display.update()
    clock.tick()


pygame.quit()
