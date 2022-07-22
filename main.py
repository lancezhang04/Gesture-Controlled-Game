import pygame
from utils.game import Player, Enemy
from utils.models import GestureRecognizer
from utils.datasets import load_configs
from utils.images import process_frame
import pandas as pd
import pickle
import cv2


# Game parameters
window_size = [600, 1000]
delay = 1
player_width, player_height = 50, 50
player_acc = 0.2  # divided by fps -> doesn't actually work
player_color = [203, 96, 21]
gesture_control = True
spawn_rate = 3000

# Initialize recognizer and capture
class_map, key_map = load_configs('configs/left_neutral_right.json')
with open('saved_models/left_neutral_right.pkl', 'rb') as f:
    clf = pickle.load(f)
recognizer = GestureRecognizer(saved_clf=clf)
capture = cv2.VideoCapture(0)

# Initialize pygame
pygame.init()
window = pygame.display.set_mode(window_size)
pygame.display.set_caption('Gesture-Controlled Game!')
clock = pygame.time.Clock()
pygame.time.set_timer(pygame.USEREVENT, spawn_rate)


run, ret = True, True
draw_fps = False
skip_frame = False  # used to increase the frame rate from 30 to 60

fps_font = pygame.font.SysFont(None, 16)
action_font = pygame.font.SysFont(None, 32)

player = Player(
    x=window_size[0] // 2 - player_width // 2,
    y=window_size[1] // 2 - player_height // 2,
    width=player_width, height=player_height,
    ax=player_acc, window_size=window_size
)

levels = pd.read_csv('levels.csv')
levels = list(levels.to_numpy())
enemies = []

while run and ret:
    pygame.time.delay(delay)

    if gesture_control:
        # Retrieve and process webcam image
        # This is where frame rate is capped
        if not skip_frame:
            ret, frame = capture.read()
            frame = process_frame(frame)

            # Model prediction and player action
            pred = recognizer.predict_image(frame)
            if pred is not None:
                if pred == 0:
                    left_pressed, right_pressed = True, False
                elif pred == 2:
                    left_pressed, right_pressed = False, True
                else:
                    left_pressed, right_pressed = False, False
                action_text = class_map[pred]
            else:
                left_pressed = right_pressed = False
                action_text = 'Neutral'
        skip_frame = not skip_frame
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
        if event.type == pygame.USEREVENT:
            # Spawn new enemy
            if len(levels) > 0:
                row = list(levels.pop(0))
                new_enemies = Enemy.create_enemies(row[1:], row[0], window_size)
                enemies.extend(new_enemies)

    # Render game
    window.fill((255, 255, 255))

    # Update and render player
    x, y = player.update_pos(clock, left_pressed, right_pressed)
    pygame.draw.rect(window, player_color, (x, y, player.width, player_height))

    # Update and render enemies; collision detection, delete if out of bond
    enemies_length = len(enemies)
    for i, enemy in enumerate(enemies[::-1]):
        pos = enemy.update_pos()
        if pos is None:
            enemies.pop(enemies_length - i - 1)
        else:
            x, y = pos
        pygame.draw.rect(window, enemy.color, (x, y, enemy.width, enemy.height))

        # Collision detection
        if enemy.rect.colliderect(player.rect):
            print('COLLISION')
            run = False

    # Draw caption for action
    action_image = action_font.render(action_text, True, player_color, (255, 255, 255))
    _, action_height = action_font.size(action_text)
    if action_image is not None:
        window.blit(action_image, (20, window_size[1] - action_height - 20))

    # Draw FPS
    if draw_fps:
        text = 'FPS: ' + str(int(clock.get_fps()))
        fps_image = fps_font.render(text, True, (255, 255, 255), (0, 0, 0))
        window.blit(fps_image, (20, 20))

    pygame.display.update()
    clock.tick()


pygame.quit()
